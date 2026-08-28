"""
MASTERCARD AI DEFENSE LAB — Phase 3: Blue Team GNN (Dual-Head Detector)
EXPERIMENT 3 — Edge Precision + Stable Thresholds

Purpose:
    Train a Graph Neural Network to detect BOTH:
    - Vector 5.1 (Mule Accounts) via Node Classification
    - Vector 5.2 (Cross-Rail Chains) via Edge Classification

Architecture:
    ┌───────────────────────────────────────────────┐
    │  SHARED BACKBONE: 3-Layer TransformerConv      │
    │  (Edge-aware attention + 3-hop receptive field) │
    └─────────┬──────────────────┬───────────────────┘
              │                  │
    ┌─────────▼────────┐ ┌──────▼─────────────┐
    │  HEAD 1 (V5.1)   │ │  HEAD 2 (V5.2)     │
    │  Node Classifier │ │  Edge Classifier    │
    │  "Is this mule?" │ │  "Is this txn       │
    │                  │ │   fraud?"            │
    │  Input: node     │ │  Input: sender_emb  │
    │  embedding       │ │  + receiver_emb     │
    │                  │ │  + edge features     │
    │                  │ │  + graph-context     │
    │                  │ │    (deg/pagerank)    │
    │  Output:         │ │  Output:            │
    │  is_mule (0/1)   │ │  is_fraud (0/1)     │
    └──────────────────┘ └────────────────────┘

Why TransformerConv over SAGEConv?
    - SAGEConv ignores edge features in message passing (simple mean aggregation)
    - TransformerConv uses edge features (amount, dwell_time, is_structured)
      directly in attention weights during message passing
    - 3 layers = 3-hop receptive field (catches longer fraud chains)
    - Multi-head attention learns WHICH neighbors matter most

EXPERIMENT 3 CHANGES (vs. Experiment 2):
    Fix 1 — Threshold search: Youden's J → F1-optimal (precision_recall_curve),
            searched on TRAIN split instead of VAL (more stable, less noisy).
    Fix 2 — Class weights capped: node ≤5x, edge ≤10x (was 38-55x → precision collapse).
    Fix 3 — Focal Loss (γ=2.0) on the EDGE head only — down-weights easy examples,
            focuses gradient on hard fraud, prevents "flag everything" behavior.
    Fix 4 — Graph-context edge features: src_out_degree, dst_in_degree,
            src_pagerank, dst_pagerank appended directly to the EDGE HEAD input
            (not the backbone attention) — gives the edge classifier explicit
            structural signal instead of relying on it being implicitly encoded
            in the 32-dim node embeddings.

Metrics tracked (per PS requirement):
    Precision, Recall, F1, AUC-ROC, False Positive Rate (FPR)
"""

import numpy as np
import pickle
import json
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================================
# NODE FEATURE COLUMN INDICES
# =====================================================================
# IMPORTANT: These indices assume node_features columns are produced in
# this exact order by compute_graph_features() (see build_graph.py):
#   ['account_age_days', 'in_degree', 'out_degree', 'pagerank',
#    'wcc_size', 'device_shared_count', 'ip_shared_count']
# This matches node_feature_names in adversarial_loop.py.
# If that column order ever changes, update these indices to match.
NODE_FEAT_IN_DEGREE = 1
NODE_FEAT_OUT_DEGREE = 2
NODE_FEAT_PAGERANK = 3


# =====================================================================
# 1. LOAD DATA & BUILD PyG DATASET
# =====================================================================

def load_gnn_dataset(path='gnn_dataset.pkl'):
    """Load the preprocessed graph data from build_graph.py"""
    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f"📂 Loaded GNN dataset:")
    print(f"   Nodes:          {data['node_features'].shape[0]}")
    print(f"   Node features:  {data['node_features'].shape[1]}")
    print(f"   Edges:          {data['edge_index'].shape[1]}")
    print(f"   Edge features:  {data['edge_features'].shape[1]}")
    print(f"   Mule ratio:     {data['node_labels'].mean()*100:.1f}%")
    print(f"   Fraud txn ratio:{data['edge_labels'].mean()*100:.2f}%")

    return data


def build_pyg_data(gnn_data):
    """Convert our dict to a PyTorch Geometric Data object with both node and edge labels."""
    import torch
    from torch_geometric.data import Data

    data = Data(
        x=torch.FloatTensor(gnn_data['node_features']),
        edge_index=torch.LongTensor(gnn_data['edge_index']),
        y=torch.LongTensor(gnn_data['node_labels']),                # Node labels (is_mule)
        edge_attr=torch.FloatTensor(gnn_data['edge_features']),      # Edge features (raw, 3-dim)
        edge_y=torch.LongTensor(gnn_data['edge_labels'])             # Edge labels (is_fraud)
    )

    num_nodes = data.num_nodes
    num_edges = data.edge_index.shape[1]

    # ─── Node Split (60/20/20) ───
    node_idx = np.random.permutation(num_nodes)
    n_train = int(0.6 * num_nodes)
    n_val = int(0.8 * num_nodes)

    node_train_mask = np.zeros(num_nodes, dtype=bool)
    node_val_mask = np.zeros(num_nodes, dtype=bool)
    node_test_mask = np.zeros(num_nodes, dtype=bool)
    node_train_mask[node_idx[:n_train]] = True
    node_val_mask[node_idx[n_train:n_val]] = True
    node_test_mask[node_idx[n_val:]] = True

    data.train_mask = torch.BoolTensor(node_train_mask)
    data.val_mask = torch.BoolTensor(node_val_mask)
    data.test_mask = torch.BoolTensor(node_test_mask)

    # ─── Edge Split (60/20/20) ───
    edge_idx = np.random.permutation(num_edges)
    e_train = int(0.6 * num_edges)
    e_val = int(0.8 * num_edges)

    edge_train_mask = np.zeros(num_edges, dtype=bool)
    edge_val_mask = np.zeros(num_edges, dtype=bool)
    edge_test_mask = np.zeros(num_edges, dtype=bool)
    edge_train_mask[edge_idx[:e_train]] = True
    edge_val_mask[edge_idx[e_train:e_val]] = True
    edge_test_mask[edge_idx[e_val:]] = True

    data.edge_train_mask = torch.BoolTensor(edge_train_mask)
    data.edge_val_mask = torch.BoolTensor(edge_val_mask)
    data.edge_test_mask = torch.BoolTensor(edge_test_mask)

    # Print stats
    print(f"\n📊 Data Split:")
    print(f"   Node Train: {node_train_mask.sum()} ({data.y[data.train_mask].sum().item()} mules)")
    print(f"   Node Val:   {node_val_mask.sum()} ({data.y[data.val_mask].sum().item()} mules)")
    print(f"   Node Test:  {node_test_mask.sum()} ({data.y[data.test_mask].sum().item()} mules)")
    print(f"   Edge Train: {edge_train_mask.sum()} ({data.edge_y[data.edge_train_mask].sum().item()} fraud)")
    print(f"   Edge Val:   {edge_val_mask.sum()} ({data.edge_y[data.edge_val_mask].sum().item()} fraud)")
    print(f"   Edge Test:  {edge_test_mask.sum()} ({data.edge_y[data.edge_test_mask].sum().item()} fraud)")

    return data


# =====================================================================
# 2. FOCAL LOSS (Fix 3 — Edge Head Only)
# =====================================================================

class FocalLoss:
    """
    Focal Loss for extreme class imbalance (Lin et al., RetinaNet, 2017).

    FL(p) = -alpha * (1 - p_t)^gamma * log(p_t)

    - Easy, confident examples (p_t close to 1) → gradient shrinks toward 0
      (don't waste capacity re-learning what's already known)
    - Hard / misclassified examples (p_t low) → gradient stays near full
      strength (model keeps learning from these)

    This directly counteracts the "flag everything as fraud" collapse that
    very high (38-55x) class weights caused in Experiment 2, because the
    loss no longer scales unboundedly with class weight alone — it also
    down-weights examples the model already handles well.

    Applied to the EDGE head only, per the Experiment 3 plan. The node
    head keeps standard weighted cross-entropy.
    """

    def __init__(self, alpha=None, gamma=2.0):
        """
        alpha: per-class weight tensor (same shape/semantics as the
               `weight` arg to F.cross_entropy), e.g. [1.0, edge_weight].
        gamma: focusing parameter. gamma=0 reduces to weighted CE.
        """
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F

        # Per-example cross-entropy (no reduction) — needed to build (1-p_t)^gamma
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')

        # p_t = probability the model assigned to the TRUE class
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        loss = focal_term * ce_loss

        return loss.mean()


# =====================================================================
# 3. DUAL-HEAD MODEL (TransformerConv — Edge-Aware Attention)
# =====================================================================

def build_model(node_in_channels, edge_in_channels, hidden_channels=64):
    """
    Dual-Head TransformerConv: Shared backbone + Node Head + Edge Head

    Backbone: 3-Layer TransformerConv (edge-aware attention, 3-hop receptive field)
    Head 1:   Linear(embedding → is_mule)
    Head 2:   MLP(sender_emb + receiver_emb + edge_features + graph_context → is_fraud)

    Why TransformerConv:
    - SAGEConv(aggr='mean') averages all neighbors equally → fraud signal diluted
    - TransformerConv uses attention + edge_attr → learns WHICH edges matter
    - edge_dim=edge_in_channels means the RAW edge features (amount, dwell_time,
      is_structured) influence attention weights during message passing

    Fix 4 (Graph-context edge features):
    The backbone's attention still only sees the raw edge_attr (edge_in_channels
    dims) — that's unchanged. But the EDGE HEAD input is now widened by 4 dims
    to explicitly include:
        src_out_degree, dst_in_degree, src_pagerank, dst_pagerank
    pulled straight from the node feature matrix `x`, so the edge classifier
    gets direct structural signal ("is the sender a hub?") instead of having
    to decode it from the 32-dim TransformerConv embedding.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import TransformerConv

    GRAPH_CONTEXT_DIM = 4  # src_out_deg, dst_in_deg, src_pagerank, dst_pagerank

    class DualHeadDetector(nn.Module):
        def __init__(self):
            super().__init__()

            # ─── Shared Backbone: 3-Layer TransformerConv ───
            # Layer 1: node features → 64, with edge attention (4 heads)
            self.conv1 = TransformerConv(
                node_in_channels, hidden_channels,
                heads=4, edge_dim=edge_in_channels, concat=False
            )
            # Layer 2: 64 → 64 (intermediate, 4 heads)
            self.conv2 = TransformerConv(
                hidden_channels, hidden_channels,
                heads=4, edge_dim=edge_in_channels, concat=False
            )
            # Layer 3: 64 → 32 (final, 1 head for clean output)
            self.conv3 = TransformerConv(
                hidden_channels, hidden_channels // 2,
                heads=1, edge_dim=edge_in_channels
            )

            # ─── Residual Connections + LayerNorm (Fix over-smoothing) ───
            # Projects input dim → hidden_channels so we can add residual
            self.input_proj = nn.Linear(node_in_channels, hidden_channels)
            self.norm1 = nn.LayerNorm(hidden_channels)
            self.norm2 = nn.LayerNorm(hidden_channels)

            self.dropout = nn.Dropout(0.3)

            emb_dim = hidden_channels // 2  # 32

            # ─── Head 1: Node Classifier (V5.1 Mule Detection) ───
            self.node_head = nn.Linear(emb_dim, 2)

            # ─── Head 2: Edge Classifier (V5.2 Cross-Rail Detection) ───
            # Input: sender_embedding(32) + receiver_embedding(32)
            #        + raw edge_features(edge_in_channels)
            #        + graph_context(4: src_out_deg, dst_in_deg, src_pr, dst_pr)
            edge_input_dim = emb_dim * 2 + edge_in_channels + GRAPH_CONTEXT_DIM
            self.edge_head = nn.Sequential(
                nn.Linear(edge_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2)
            )

        def get_embeddings(self, x, edge_index, edge_attr=None):
            """Extract node embeddings from 3-layer TransformerConv backbone.
            
            Uses Residual (Skip) Connections + LayerNorm to prevent
            over-smoothing on highly connected topologies (fan_out_fan_in).
            """
            # Project input to hidden_channels for residual addition
            identity = self.input_proj(x)

            # Layer 1 + Residual
            h = self.conv1(x, edge_index, edge_attr)
            h = self.norm1(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + identity  # Skip connection: preserves 1-hop signal

            # Layer 2 + Residual
            identity2 = h
            h = self.conv2(h, edge_index, edge_attr)
            h = self.norm2(h)
            h = F.relu(h)
            h = self.dropout(h)
            h = h + identity2  # Skip connection: preserves 2-hop signal

            # Layer 3 (dimension change 64→32, no residual)
            h = self.conv3(h, edge_index, edge_attr)
            h = F.relu(h)
            return h

        def forward(self, x, edge_index, edge_attr=None):
            """
            Forward pass through both heads.

            Returns:
                node_out: (num_nodes, 2) — mule classification logits
                edge_out: (num_edges, 2) — fraud classification logits
            """
            # Backbone: get node embeddings (raw edge_attr used in attention)
            embeddings = self.get_embeddings(x, edge_index, edge_attr)

            # Head 1: Node classification
            node_out = self.node_head(embeddings)

            # Head 2: Edge classification
            src_idx = edge_index[0]  # Sender node indices
            dst_idx = edge_index[1]  # Receiver node indices

            src_emb = embeddings[src_idx]   # Sender embeddings
            dst_emb = embeddings[dst_idx]   # Receiver embeddings

            # ─── Fix 4: Graph-context features (explicit structural signal) ───
            src_out_deg = x[src_idx, NODE_FEAT_OUT_DEGREE:NODE_FEAT_OUT_DEGREE + 1]
            dst_in_deg = x[dst_idx, NODE_FEAT_IN_DEGREE:NODE_FEAT_IN_DEGREE + 1]
            src_pagerank = x[src_idx, NODE_FEAT_PAGERANK:NODE_FEAT_PAGERANK + 1]
            dst_pagerank = x[dst_idx, NODE_FEAT_PAGERANK:NODE_FEAT_PAGERANK + 1]
            graph_context = torch.cat(
                [src_out_deg, dst_in_deg, src_pagerank, dst_pagerank], dim=1
            )

            # Concatenate: [sender_emb | receiver_emb | edge_features | graph_context]
            if edge_attr is not None:
                edge_input = torch.cat([src_emb, dst_emb, edge_attr, graph_context], dim=1)
            else:
                zeros = torch.zeros(src_emb.shape[0], edge_in_channels, device=src_emb.device)
                edge_input = torch.cat([src_emb, dst_emb, zeros, graph_context], dim=1)

            edge_out = self.edge_head(edge_input)

            return node_out, edge_out

    model = DualHeadDetector()
    print(f"\n🧠 Model Architecture:")
    print(f"   {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    return model


# =====================================================================
# 4. TRAINING LOOP (Joint Training)
# =====================================================================

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4,
                 node_weight_cap=5.0, edge_weight_cap=10.0, focal_gamma=2.0):
    """
    Train both heads jointly.

    Fix 2 — Class weights are now CAPPED (node ≤5x, edge ≤10x) instead of
    the raw n_legit/n_fraud ratio (which hit 38-55x and caused the model to
    over-flag "when in doubt, it's fraud").

    Fix 3 — The EDGE head uses Focal Loss (gamma=2.0) instead of plain
    weighted cross-entropy, so the loss focuses on hard/misclassified
    examples rather than uniformly amplifying every fraud example.
    The NODE head keeps standard weighted cross-entropy.

    Total Loss = node_loss (weighted CE) + edge_loss (focal)
    """
    import torch
    import torch.nn.functional as F

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ─── Node Class Weights (capped) ───
    n_legit_nodes = (data.y[data.train_mask] == 0).sum().item()
    n_mule_nodes = max((data.y[data.train_mask] == 1).sum().item(), 1)
    raw_node_ratio = n_legit_nodes / n_mule_nodes
    node_weights = torch.FloatTensor([1.0, min(node_weight_cap, raw_node_ratio)])

    # ─── Edge Class Weights (capped) ───
    n_legit_edges = (data.edge_y[data.edge_train_mask] == 0).sum().item()
    n_fraud_edges = max((data.edge_y[data.edge_train_mask] == 1).sum().item(), 1)
    raw_edge_ratio = n_legit_edges / n_fraud_edges
    edge_weights = torch.FloatTensor([1.0, min(edge_weight_cap, raw_edge_ratio)])

    edge_focal_loss = FocalLoss(alpha=edge_weights, gamma=focal_gamma)

    print(f"\n⚖️  Class Weights (capped: node≤{node_weight_cap}x, edge≤{edge_weight_cap}x):")
    print(f"   Node: Legit={node_weights[0]:.2f}, Mule={node_weights[1]:.2f} (raw ratio: {raw_node_ratio:.1f}x)")
    print(f"   Edge: Legit={edge_weights[0]:.2f}, Fraud={edge_weights[1]:.2f} (raw ratio: {raw_edge_ratio:.1f}x)")
    print(f"   Edge loss: Focal Loss (gamma={focal_gamma})")
    print(f"\n🔄 Training for {epochs} epochs...\n")

    best_val_score = 0.0
    best_model_state = None
    patience = 60
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        node_out, edge_out = model(data.x, data.edge_index, data.edge_attr)

        # Node loss (Head 1) — standard weighted cross-entropy
        node_loss = F.cross_entropy(
            node_out[data.train_mask],
            data.y[data.train_mask],
            weight=node_weights
        )

        # Edge loss (Head 2) — Focal Loss (Fix 3)
        edge_loss = edge_focal_loss(
            edge_out[data.edge_train_mask],
            data.edge_y[data.edge_train_mask]
        )

        # Joint loss
        total_loss = node_loss + edge_loss
        total_loss.backward()
        optimizer.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            val_node = evaluate_nodes(model, data, 'val')
            val_edge = evaluate_edges(model, data, 'val')

            # Combined score for early stopping
            combined_f1 = (val_node['f1'] + val_edge['f1']) / 2

            print(f"   Epoch {epoch+1:3d} | Loss: {total_loss.item():.4f} | "
                  f"Node F1: {val_node['f1']:.3f} | Edge F1: {val_edge['f1']:.3f} | "
                  f"Combined: {combined_f1:.3f}")

            if combined_f1 > best_val_score:
                best_val_score = combined_f1
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 10

            if no_improve >= patience:
                print(f"\n   ⏹️  Early stopping at epoch {epoch+1}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n   ✅ Loaded best model (Combined F1: {best_val_score:.4f})")

    return model


# =====================================================================
# 5. EVALUATION FUNCTIONS (Full PS Metrics + F1-Optimal Threshold)
# =====================================================================

def find_optimal_threshold(model, data, head='node', search_split='train'):
    """
    Find optimal decision threshold that MAXIMIZES F1 (Fix 1).

    Previously used Youden's J (argmax TPR - FPR) on the VALIDATION set,
    which was unstable because val only has 7-20 fraud edges → extremely
    noisy ROC curve → thresholds swinging between 0.30 (floor) and 0.90
    across epochs. Also, Youden's J optimizes TPR-FPR, not F1, which is
    the metric we actually care about.

    Fix: use sklearn's precision_recall_curve + argmax(F1), searched on
    the TRAIN split (1200 nodes >> 400 val nodes → much more stable).

    Returns: optimal threshold (float, 0.0-1.0)
    """
    import torch
    from sklearn.metrics import precision_recall_curve

    model.eval()
    with torch.no_grad():
        node_out, edge_out = model(data.x, data.edge_index, data.edge_attr)

    if head == 'node':
        mask = getattr(data, f'{search_split}_mask')
        y_true = data.y[mask].cpu().numpy()
        probs = torch.softmax(node_out, dim=1)[:, 1][mask].detach().cpu().numpy()
    else:
        mask = getattr(data, f'edge_{search_split}_mask')
        y_true = data.edge_y[mask].cpu().numpy()
        probs = torch.softmax(edge_out, dim=1)[:, 1][mask].detach().cpu().numpy()

    # Need both classes present to search a threshold
    if len(set(y_true)) < 2:
        return 0.5

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, probs)

    # F1 at each threshold: precision_recall_curve returns len(thresholds) == len(precision_arr) - 1
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    # Drop the last precision/recall point (threshold=inf, no corresponding threshold value)
    f1_scores = f1_scores[:-1]

    if len(f1_scores) == 0:
        return 0.5

    best_idx = f1_scores.argmax()
    optimal_threshold = float(thresholds[best_idx])

    return optimal_threshold


def evaluate_nodes(model, data, split='test', threshold=None):
    """
    Evaluate Head 1 (Node Classification — Mule Detection).

    If threshold is provided, uses it instead of argmax (0.5).
    This allows precision/FPR optimization without retraining.

    Returns ALL PS-required metrics:
    Precision, Recall, F1, AUC-ROC, PR-AUC, FPR, confusion matrix
    """
    import torch
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, roc_auc_score,
                                  average_precision_score)

    model.eval()
    mask = getattr(data, f'{split}_mask') if split != 'edge_test' else data.test_mask

    with torch.no_grad():
        node_out, _ = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(node_out, dim=1)[:, 1].detach().cpu()  # .detach() safety

    y_true = data.y[mask].cpu().numpy()
    y_probs = probs[mask].numpy()

    # Apply threshold: custom or default 0.5
    if threshold is not None:
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = node_out.argmax(dim=1)[mask].cpu().numpy()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # False Positive Rate with epsilon safety
    fpr = fp / (fp + tn + 1e-8)

    # AUC-ROC (needs both classes present)
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0

    # PR-AUC (better than ROC-AUC for imbalanced data)
    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except ValueError:
        pr_auc = 0.0

    return {
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc': round(auc, 4),
        'pr_auc': round(pr_auc, 4),
        'fpr': round(fpr, 4),
        'threshold': round(threshold, 4) if threshold else 0.5,
        'confusion_matrix': cm.tolist(),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'total': int(mask.sum()),
        'actual_positive': int((y_true == 1).sum()),
        'predicted_positive': int((y_pred == 1).sum())
    }


def evaluate_edges(model, data, split='test', threshold=None):
    """
    Evaluate Head 2 (Edge Classification — Fraud Transaction Detection).

    If threshold is provided, uses it instead of argmax (0.5).

    Returns ALL PS-required metrics:
    Precision, Recall, F1, AUC-ROC, PR-AUC, FPR, confusion matrix
    """
    import torch
    from sklearn.metrics import (precision_score, recall_score, f1_score,
                                  confusion_matrix, roc_auc_score,
                                  average_precision_score)

    model.eval()
    mask = getattr(data, f'edge_{split}_mask')

    with torch.no_grad():
        _, edge_out = model(data.x, data.edge_index, data.edge_attr)
        probs = torch.softmax(edge_out, dim=1)[:, 1].detach().cpu()  # .detach() safety

    y_true = data.edge_y[mask].cpu().numpy()
    y_probs = probs[mask].numpy()

    # Apply threshold: custom or default 0.5
    if threshold is not None:
        y_pred = (y_probs >= threshold).astype(int)
    else:
        y_pred = edge_out.argmax(dim=1)[mask].cpu().numpy()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # False Positive Rate with epsilon safety
    fpr = fp / (fp + tn + 1e-8)

    # AUC-ROC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0

    # PR-AUC
    try:
        pr_auc = average_precision_score(y_true, y_probs)
    except ValueError:
        pr_auc = 0.0

    return {
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'auc_roc': round(auc, 4),
        'pr_auc': round(pr_auc, 4),
        'fpr': round(fpr, 4),
        'threshold': round(threshold, 4) if threshold else 0.5,
        'confusion_matrix': cm.tolist(),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'total': int(mask.sum()),
        'actual_positive': int((y_true == 1).sum()),
        'predicted_positive': int((y_pred == 1).sum())
    }


def print_final_report(node_metrics, edge_metrics):
    """Print comprehensive dual-head evaluation report with ALL PS metrics."""

    print(f"\n{'='*60}")
    print(f"  BLUE TEAM — DUAL-HEAD EVALUATION REPORT")
    print(f"  (PS Metrics: Precision, Recall, F1, AUC-ROC, PR-AUC, FPR)")
    print(f"{'='*60}")

    for head_name, splits in [("HEAD 1: MULE DETECTION (V5.1)", node_metrics),
                                ("HEAD 2: FRAUD TXN DETECTION (V5.2)", edge_metrics)]:
        print(f"\n  {'─'*55}")
        print(f"  {head_name}")
        print(f"  {'─'*55}")

        for split_name, m in splits.items():
            print(f"\n    {split_name.upper()} ({m['total']} samples, {m['actual_positive']} positive):")
            print(f"      Precision:  {m['precision']:.4f}")
            print(f"      Recall:     {m['recall']:.4f}")
            print(f"      F1:         {m['f1']:.4f}")
            print(f"      AUC-ROC:    {m['auc_roc']:.4f}")
            print(f"      PR-AUC:     {m.get('pr_auc', 0):.4f}")
            print(f"      FPR:        {m['fpr']:.4f}  {'✅' if m['fpr'] < 0.05 else '⚠️ HIGH'}")
            print(f"      Threshold:  {m.get('threshold', 0.5):.4f}")
            print(f"      Confusion:  TN={m['tn']} FP={m['fp']} FN={m['fn']} TP={m['tp']}")

    # Overall verdict
    test_node = node_metrics['test']
    test_edge = edge_metrics['test']
    combined_f1 = (test_node['f1'] + test_edge['f1']) / 2
    combined_auc = (test_node['auc_roc'] + test_edge['auc_roc']) / 2

    print(f"\n  {'='*60}")
    print(f"  📊 FINAL SCORES (PS-ALIGNED):")
    print(f"     {'Metric':<20} {'V5.1 Mule':<12} {'V5.2 Fraud':<12} {'Target'}")
    print(f"     {'─'*56}")
    print(f"     {'Precision':<20} {test_node['precision']:<12.4f} {test_edge['precision']:<12.4f} >0.80")
    print(f"     {'Recall':<20} {test_node['recall']:<12.4f} {test_edge['recall']:<12.4f} >0.75")
    print(f"     {'F1':<20} {test_node['f1']:<12.4f} {test_edge['f1']:<12.4f} >0.80")
    print(f"     {'AUC-ROC':<20} {test_node['auc_roc']:<12.4f} {test_edge['auc_roc']:<12.4f} >0.85")
    print(f"     {'PR-AUC':<20} {test_node.get('pr_auc',0):<12.4f} {test_edge.get('pr_auc',0):<12.4f} >0.70")
    print(f"     {'FPR':<20} {test_node['fpr']:<12.4f} {test_edge['fpr']:<12.4f} <0.05")
    print(f"     {'Threshold':<20} {test_node.get('threshold',0.5):<12.4f} {test_edge.get('threshold',0.5):<12.4f} F1-Optimal")
    print(f"     {'─'*56}")
    print(f"     Combined F1:  {combined_f1:.4f}")
    print(f"     Combined AUC: {combined_auc:.4f}")
    print(f"  {'='*60}")


# =====================================================================
# 6. MAIN PIPELINE
# =====================================================================

if __name__ == "__main__":
    import torch
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("  MASTERCARD AI DEFENSE LAB — BLUE TEAM GNN")
    print("  Phase 3: Dual-Head Detector (V5.1 + V5.2) — Experiment 3")
    print("=" * 60)

    # ─── Step 1: Load Data ───
    gnn_data = load_gnn_dataset()

    # ─── Step 2: Build PyG Data ───
    data = build_pyg_data(gnn_data)

    # ─── Step 3: Build Dual-Head Model ───
    node_in = data.x.shape[1]       # 7 node features
    edge_in = data.edge_attr.shape[1]  # 3 raw edge features (backbone attention)
    model = build_model(node_in_channels=node_in, edge_in_channels=edge_in, hidden_channels=64)

    # ─── Step 4: Train ───
    model = train_model(model, data, epochs=200, lr=0.01)

    # ─── Step 5: Find F1-optimal thresholds (Fix 1) & Evaluate Both Heads ───
    node_threshold = find_optimal_threshold(model, data, head='node', search_split='train')
    edge_threshold = find_optimal_threshold(model, data, head='edge', search_split='train')
    print(f"\n  🎯 F1-Optimal Thresholds (searched on TRAIN split):")
    print(f"     Node: {node_threshold:.4f} (default was 0.50)")
    print(f"     Edge: {edge_threshold:.4f} (default was 0.50)")

    node_metrics = {
        'train': evaluate_nodes(model, data, 'train', threshold=node_threshold),
        'val': evaluate_nodes(model, data, 'val', threshold=node_threshold),
        'test': evaluate_nodes(model, data, 'test', threshold=node_threshold)
    }

    edge_metrics = {
        'train': evaluate_edges(model, data, 'train', threshold=edge_threshold),
        'val': evaluate_edges(model, data, 'val', threshold=edge_threshold),
        'test': evaluate_edges(model, data, 'test', threshold=edge_threshold)
    }

    print_final_report(node_metrics, edge_metrics)

    # ─── Step 6: Save ───
    print(f"\n💾 Saving model and metrics...")

    torch.save(model.state_dict(), "blue_team_model.pt")
    print(f"   ✅ Model saved: blue_team_model.pt")

    baseline = {
        'node_metrics': node_metrics,
        'edge_metrics': edge_metrics,
        'timestamp': datetime.now().isoformat(),
        'model_type': 'DualHead_TransformerConv_Exp3',
        'architecture': {
            'backbone': 'TransformerConv_3Layer_EdgeAware',
            'head1': 'NodeClassifier (V5.1 Mules) — weighted CE, capped 5x',
            'head2': 'EdgeClassifier (V5.2 Cross-Rail) — Focal Loss (gamma=2.0), capped 10x, +graph-context features',
            'node_features': int(node_in),
            'edge_features_raw': int(edge_in),
            'edge_head_graph_context_dims': 4,
            'hidden_channels': 64,
            'attention_heads': '4/4/1',
            'receptive_field': '3-hop',
            'threshold_method': 'F1-optimal (precision_recall_curve, searched on train split)'
        }
    }

    with open("baseline_metrics.json", "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"   ✅ Metrics saved: baseline_metrics.json")

    print(f"\n  Next: Phase 4 → dynamic_attack_catalog.py + adversarial_loop.py")