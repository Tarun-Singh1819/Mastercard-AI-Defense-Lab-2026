"""
MASTERCARD AI DEFENSE LAB — Phase 3: Blue Team GNN (Dual-Head Detector)
EXPERIMENT HGT — Heterogeneous Graph Transformer

Purpose:
    Train a Heterogeneous Graph Neural Network to detect:
    - Vector 5.1 (Mule Accounts) via Node Classification
    - Vector 5.2 (Cross-Rail Chains) via Edge Classification

Architecture:
    Uses HGTConv to natively understand the difference between:
    - Account nodes
    - Device nodes
    - IP nodes
    And different edge types:
    - (Account, transfer, Account)
    - (Account, uses, Device)
    - (Account, uses, IP)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear

# =====================================================================
# NODE FEATURE COLUMN INDICES
# =====================================================================
NODE_FEAT_IN_DEGREE = 1
NODE_FEAT_OUT_DEGREE = 2
NODE_FEAT_PAGERANK = 3

# =====================================================================
# 1. LOAD DATA & BUILD PyG DATASET
# =====================================================================

def build_pyg_data(gnn_data):
    """Convert our dict to a PyTorch Geometric HeteroData object."""
    data = HeteroData()

    # --- Node Features & Labels ---
    data['account'].x = torch.FloatTensor(gnn_data['node_features_account'])
    data['account'].y = torch.LongTensor(gnn_data['node_labels_account'])
    
    # Device and IP nodes don't have natural features, so we use dummy 1s.
    # HGT requires node features.
    num_devices = gnn_data['num_nodes_device']
    num_ips = gnn_data['num_nodes_ip']
    
    # Give them a simple 1D constant feature
    data['device'].x = torch.ones((num_devices, 1))
    data['ip'].x = torch.ones((num_ips, 1))
    
    # --- Edges ---
    # Transfer edges (Account -> Account)
    data['account', 'transfer', 'account'].edge_index = torch.LongTensor(gnn_data['edge_index_transfer'])
    data['account', 'transfer', 'account'].edge_attr = torch.FloatTensor(gnn_data['edge_features_transfer'])
    data['account', 'transfer', 'account'].edge_y = torch.LongTensor(gnn_data['edge_labels_transfer'])
    
    # Uses Device edges (Account -> Device)
    data['account', 'uses', 'device'].edge_index = torch.LongTensor(gnn_data['edge_index_uses_device'])
    
    # Uses IP edges (Account -> IP)
    data['account', 'uses', 'ip'].edge_index = torch.LongTensor(gnn_data['edge_index_uses_ip'])
    
    # Add reverse edges for message passing in both directions
    # Device -> Account
    rev_dev = data['account', 'uses', 'device'].edge_index.flip([0])
    data['device', 'used_by', 'account'].edge_index = rev_dev
    
    # IP -> Account
    rev_ip = data['account', 'uses', 'ip'].edge_index.flip([0])
    data['ip', 'used_by', 'account'].edge_index = rev_ip

    num_acc_nodes = data['account'].num_nodes
    num_edges = data['account', 'transfer', 'account'].edge_index.shape[1]

    # ─── Node Split (60/20/20) ───
    node_idx = np.random.permutation(num_acc_nodes)
    n_train = int(0.6 * num_acc_nodes)
    n_val = int(0.8 * num_acc_nodes)

    node_train_mask = np.zeros(num_acc_nodes, dtype=bool)
    node_val_mask = np.zeros(num_acc_nodes, dtype=bool)
    node_test_mask = np.zeros(num_acc_nodes, dtype=bool)
    node_train_mask[node_idx[:n_train]] = True
    node_val_mask[node_idx[n_train:n_val]] = True
    node_test_mask[node_idx[n_val:]] = True

    data['account'].train_mask = torch.BoolTensor(node_train_mask)
    data['account'].val_mask = torch.BoolTensor(node_val_mask)
    data['account'].test_mask = torch.BoolTensor(node_test_mask)

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

    transfer_edge = ('account', 'transfer', 'account')
    data[transfer_edge].train_mask = torch.BoolTensor(edge_train_mask)
    data[transfer_edge].val_mask = torch.BoolTensor(edge_val_mask)
    data[transfer_edge].test_mask = torch.BoolTensor(edge_test_mask)

    return data


# =====================================================================
# 2. FOCAL LOSS
# =====================================================================

class FocalLoss:
    def __init__(self, alpha=None, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        probs = torch.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - p_t).clamp(min=1e-8) ** self.gamma
        loss = focal_term * ce_loss
        return loss.mean()


# =====================================================================
# 3. DUAL-HEAD HGT MODEL
# =====================================================================

def build_model(data, hidden_channels=64, num_heads=4, num_layers=2):
    """
    Dual-Head HGT: 
    Backbone: HGTConv layers (Heterogeneous Graph Transformer)
    Head 1:   Linear(account_emb → is_mule)
    Head 2:   MLP(sender_emb + receiver_emb + edge_features + graph_context → is_fraud)
    """
    GRAPH_CONTEXT_DIM = 4 

    class DualHeadHGTDetector(nn.Module):
        def __init__(self, metadata, edge_in_channels):
            super().__init__()
            
            # Since different nodes have different input feature dimensions,
            # HGT needs them all projected to the same hidden_channels first.
            self.lin_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                in_channels = data[node_type].x.size(1)
                self.lin_dict[node_type] = Linear(in_channels, hidden_channels)

            self.convs = nn.ModuleList()
            for _ in range(num_layers):
                conv = HGTConv(hidden_channels, hidden_channels, metadata,
                               num_heads)
                self.convs.append(conv)
                
            self.dropout = nn.Dropout(0.3)

            # Node Head (Account Mule Detection)
            self.node_head = nn.Linear(hidden_channels, 2)

            # Edge Head (Cross-Rail Transfer Detection)
            edge_input_dim = hidden_channels * 2 + edge_in_channels + GRAPH_CONTEXT_DIM
            self.edge_head = nn.Sequential(
                nn.Linear(edge_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 2)
            )

        def get_embeddings(self, x_dict, edge_index_dict):
            # Project all node features to hidden_channels
            x_dict_proj = {
                node_type: self.lin_dict[node_type](x).relu_()
                for node_type, x in x_dict.items()
            }
            
            # Apply HGT layers
            for conv in self.convs:
                x_dict_proj = conv(x_dict_proj, edge_index_dict)
                # Apply relu/dropout to all node types
                x_dict_proj = {k: self.dropout(F.relu(v)) for k, v in x_dict_proj.items()}
                
            return x_dict_proj

        def forward(self, x_dict, edge_index_dict, transfer_edge_attr, raw_account_x):
            embeddings = self.get_embeddings(x_dict, edge_index_dict)
            account_emb = embeddings['account']
            
            # Node classification (only on account nodes)
            node_out = self.node_head(account_emb)
            
            # Edge classification (only on 'transfer' edges)
            transfer_edge_index = edge_index_dict[('account', 'transfer', 'account')]
            src_idx = transfer_edge_index[0]
            dst_idx = transfer_edge_index[1]
            
            src_emb = account_emb[src_idx]
            dst_emb = account_emb[dst_idx]
            
            # Graph context directly from original features
            src_out_deg = raw_account_x[src_idx, NODE_FEAT_OUT_DEGREE:NODE_FEAT_OUT_DEGREE + 1]
            dst_in_deg = raw_account_x[dst_idx, NODE_FEAT_IN_DEGREE:NODE_FEAT_IN_DEGREE + 1]
            src_pagerank = raw_account_x[src_idx, NODE_FEAT_PAGERANK:NODE_FEAT_PAGERANK + 1]
            dst_pagerank = raw_account_x[dst_idx, NODE_FEAT_PAGERANK:NODE_FEAT_PAGERANK + 1]
            
            graph_context = torch.cat(
                [src_out_deg, dst_in_deg, src_pagerank, dst_pagerank], dim=1
            )
            
            edge_input = torch.cat([src_emb, dst_emb, transfer_edge_attr, graph_context], dim=1)
            edge_out = self.edge_head(edge_input)
            
            return node_out, edge_out

    transfer_edge_attr_dim = data['account', 'transfer', 'account'].edge_attr.size(1)
    model = DualHeadHGTDetector(data.metadata(), transfer_edge_attr_dim)
    
    print(f"\n🧠 HGT Model Architecture:")
    print(f"   {model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    return model


# =====================================================================
# 4. TRAINING LOOP
# =====================================================================

def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4,
                 node_weight_cap=5.0, edge_weight_cap=10.0, focal_gamma=2.0):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ─── Node Class Weights ───
    train_mask = data['account'].train_mask
    labels = data['account'].y
    n_legit_nodes = (labels[train_mask] == 0).sum().item()
    n_mule_nodes = max((labels[train_mask] == 1).sum().item(), 1)
    raw_node_ratio = n_legit_nodes / n_mule_nodes
    node_weights = torch.FloatTensor([1.0, min(node_weight_cap, raw_node_ratio)])

    # ─── Edge Class Weights ───
    transfer_edge = ('account', 'transfer', 'account')
    e_train_mask = data[transfer_edge].train_mask
    e_labels = data[transfer_edge].edge_y
    n_legit_edges = (e_labels[e_train_mask] == 0).sum().item()
    n_fraud_edges = max((e_labels[e_train_mask] == 1).sum().item(), 1)
    raw_edge_ratio = n_legit_edges / n_fraud_edges
    edge_weights = torch.FloatTensor([1.0, min(edge_weight_cap, raw_edge_ratio)])

    edge_focal_loss = FocalLoss(alpha=edge_weights, gamma=focal_gamma)

    print(f"\n⚖️  Class Weights (capped: node≤{node_weight_cap}x, edge≤{edge_weight_cap}x):")
    print(f"   Node: Legit={node_weights[0]:.2f}, Mule={node_weights[1]:.2f} (raw ratio: {raw_node_ratio:.1f}x)")
    print(f"   Edge: Legit={edge_weights[0]:.2f}, Fraud={edge_weights[1]:.2f} (raw ratio: {raw_edge_ratio:.1f}x)")

    best_val_score = 0.0
    best_model_state = None
    patience = 60
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr = data[transfer_edge].edge_attr
        raw_account_x = data['account'].x
        
        node_out, edge_out = model(x_dict, edge_index_dict, edge_attr, raw_account_x)

        node_loss = F.cross_entropy(
            node_out[train_mask],
            labels[train_mask],
            weight=node_weights
        )

        edge_loss = edge_focal_loss(
            edge_out[e_train_mask],
            e_labels[e_train_mask]
        )

        total_loss = node_loss + edge_loss
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            val_node = evaluate_nodes(model, data, 'val')
            val_edge = evaluate_edges(model, data, 'val')
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
# 5. EVALUATION FUNCTIONS
# =====================================================================

def find_optimal_threshold(model, data, head='node', search_split='train'):
    from sklearn.metrics import precision_recall_curve
    model.eval()
    with torch.no_grad():
        node_out, edge_out = model(data.x_dict, data.edge_index_dict, 
                                   data[('account', 'transfer', 'account')].edge_attr,
                                   data['account'].x)

    if head == 'node':
        mask = data['account'][f'{search_split}_mask']
        y_true = data['account'].y[mask].cpu().numpy()
        probs = torch.softmax(node_out, dim=1)[:, 1][mask].detach().cpu().numpy()
    else:
        transfer_edge = ('account', 'transfer', 'account')
        mask = data[transfer_edge][f'{search_split}_mask']
        y_true = data[transfer_edge].edge_y[mask].cpu().numpy()
        probs = torch.softmax(edge_out, dim=1)[:, 1][mask].detach().cpu().numpy()

    if len(set(y_true)) < 2:
        return 0.5

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)
    f1_scores = f1_scores[:-1]

    if len(f1_scores) == 0:
        return 0.5

    best_idx = f1_scores.argmax()
    return float(thresholds[best_idx])

def evaluate_nodes(model, data, split='test', threshold=None):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
    model.eval()
    with torch.no_grad():
        node_out, _ = model(data.x_dict, data.edge_index_dict, 
                            data[('account', 'transfer', 'account')].edge_attr,
                            data['account'].x)

    mask = data['account'][f'{split}_mask']
    y_true = data['account'].y[mask].cpu().numpy()
    probs = torch.softmax(node_out, dim=1)[:, 1][mask].detach().cpu().numpy()

    if threshold is None:
        threshold = 0.5
    y_pred = (probs >= threshold).astype(int)

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
    else:
        auc = 0.0
        pr_auc = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1, 
            'auc': auc, 'pr_auc': pr_auc, 'fpr': fpr, 'threshold': threshold}

def evaluate_edges(model, data, split='test', threshold=None):
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, average_precision_score
    model.eval()
    with torch.no_grad():
        _, edge_out = model(data.x_dict, data.edge_index_dict, 
                            data[('account', 'transfer', 'account')].edge_attr,
                            data['account'].x)

    transfer_edge = ('account', 'transfer', 'account')
    mask = data[transfer_edge][f'{split}_mask']
    y_true = data[transfer_edge].edge_y[mask].cpu().numpy()
    probs = torch.softmax(edge_out, dim=1)[:, 1][mask].detach().cpu().numpy()

    if threshold is None:
        threshold = 0.5
    y_pred = (probs >= threshold).astype(int)

    if len(set(y_true)) > 1:
        auc = roc_auc_score(y_true, probs)
        pr_auc = average_precision_score(y_true, probs)
    else:
        auc = 0.0
        pr_auc = 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'auc': auc, 'pr_auc': pr_auc, 'fpr': fpr, 'threshold': threshold}
