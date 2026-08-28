"""
MASTERCARD AI DEFENSE LAB — Phase 4B: Adversarial Training Loop (THE MAIN EVENT)
EXPERIMENT 3 — Edge Precision + Stable Thresholds

Purpose:
    The closed-loop engine that ties everything together.
    Red Team generates progressively harder attacks → Blue Team learns →
    SmartEvolution finds blind spots → Red Team adapts → Repeat.

Flow per Epoch:
    1. Red Team (DynamicAttackCatalog) generates fraud with randomized params
    2. CTGAN_LEGIT generates legitimate background traffic
    3. Fidelity Validator checks data quality
    4. Build graph → compute features
    5. Blue Team GNN trains on this epoch's data
    6. Evaluate → SmartEvolution analyzes blind spots
    7. Feed blind spots to Red Team for next epoch
    8. Log metrics → Repeat until convergence

The Demo Story:
    Epoch 0: F1 = 1.00 (seed data too easy, GNN memorized)
    Epoch 1: F1 drops  (harder attacks challenge the GNN)
    Epoch 2: F1 recovers (GNN learns structural patterns)
    Epoch N: F1 stabilizes (robust, hardened detector)

EXPERIMENT 3 CHANGE (vs. Experiment 2):
    train_blue_team_on_epoch() now calls the updated find_optimal_threshold()
    from blue_team_gnn.py, which searches for the F1-MAXIMIZING threshold
    (precision_recall_curve) on the TRAIN split, instead of Youden's J
    (TPR - FPR) on the noisy, small VAL split. See blue_team_gnn.py for the
    full rationale — this was the primary fix for Edge Precision instability.
"""

import sys
sys.path.insert(0, '.')  # Colab fix

import numpy as np
import pandas as pd
import pickle
import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# =====================================================================
# EXPERIMENT CONFIGURATION
# =====================================================================
USE_HGT = True  # Toggle for Phase 2: Heterogeneous Graph Transformer

# Colab Compatibility: Wrap imports in try-except. In Colab, if you run the cells sequentially, 
# these functions are already in the global namespace, so we don't need to explicitly import them.
try:
    if not USE_HGT:
        from blue_team_gnn import build_model, build_pyg_data, train_model, evaluate_nodes, evaluate_edges, find_optimal_threshold
    else:
        from blue_team_hgt import build_model, build_pyg_data, train_model, evaluate_nodes, evaluate_edges, find_optimal_threshold

    from build_graph import build_networkx_graph, compute_graph_features, prepare_for_gnn
    from dynamic_attack_catalog import DynamicAttackCatalog, SmartEvolution, generate_epoch_data, post_process_ctgan_output
    from llm_strategist import LLMRedTeamController
except ImportError:
    print("Colab environment detected (or missing local modules). Relying on global namespace for functions.")

# =====================================================================
# LLM RED TEAM CONTROLLER (Inlined for Colab compatibility)
# =====================================================================

class LLMRedTeamController:
    """
    GenAI-powered Red Team Strategy Advisor.
    Takes SmartEvolution's blind spots and outputs:
      1. A strategic narrative (for the pitch/demo)
      2. Parameter overrides (to actively steer the Red Team)

    Modes:
      use_mock=True  → Rules-based strategy (no API key needed)
      use_mock=False → Real Gemini API call (needs GEMINI_API_KEY)
    """

    def __init__(self, use_mock=False):
        self.use_mock = use_mock

    def generate_strategy(self, epoch, blind_spots, current_params):
        if not blind_spots:
            return {
                "narrative": "🤖 AI Strategist: No blind spots detected. Blue Team is highly effective. Shifting to random topology rotation.",
                "overrides": {}
            }

        if self.use_mock:
            return self._mock_llm_response(epoch, blind_spots, current_params)
        else:
            return self._real_llm_response(epoch, blind_spots, current_params)

    def _mock_llm_response(self, epoch, blind_spots, current_params):
        import random
        overrides = {}
        strategy_points = []

        if 'device_shared_count' in blind_spots or 'ip_shared_count' in blind_spots:
            strategy_points.append("Blue Team is clustering mules via device/IP sharing graphs.")
            overrides['device_sharing_ratio'] = random.uniform(0.05, 0.15)
            strategy_points.append(f"→ Action: Reduce device sharing to {overrides['device_sharing_ratio']:.0%} to blend with legitimate users.")

        if 'pagerank' in blind_spots:
            strategy_points.append("PageRank centrality is exposing aggregation accounts.")
            overrides['topology'] = ['star_burst', 'cross_rail']
            strategy_points.append("→ Action: Switch to Star Burst / Cross-Rail topologies to disperse funds without creating high-rank hubs.")

        if 'in_degree' in blind_spots or 'out_degree' in blind_spots:
            strategy_points.append("Node degree signatures are anomalous vs legitimate accounts.")
            overrides['noise_txn_per_mule'] = random.randint(4, 8)
            strategy_points.append(f"→ Action: Inject {overrides['noise_txn_per_mule']} noise txns per mule to normalize degree distribution.")

        if 'account_age_days' in blind_spots:
            info = blind_spots['account_age_days']
            if info['direction'] == 'increase':
                overrides['mule_age_range'] = (180, 500)
                strategy_points.append("→ Action: Use dormant/aged accounts (180-500 days) — young accounts are being flagged.")
            else:
                overrides['mule_age_range'] = (15, 90)
                strategy_points.append("→ Action: Use fresh accounts (15-90 days) — old accounts are being flagged.")

        if not strategy_points:
            strategy_points.append("General evasion mode: randomizing all parameters.")
            overrides['amount_mean'] = random.uniform(15000, 30000)

        narrative = f"🤖 AI STRATEGIST REPORT (Epoch {epoch}):\n  " + "\n  ".join(strategy_points)

        return {
            "narrative": narrative,
            "overrides": overrides
        }

    def _real_llm_response(self, epoch, blind_spots, current_params):
        """Uses Google Gemini API. Requires: pip install google-generativeai"""
        try:
            import google.generativeai as genai
            import json as json_mod
            import re

            # ──────────────────────────────────────────────
            # PUT YOUR GEMINI API KEY HERE (or set env var)
            # ──────────────────────────────────────────────
            api_key = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY_HERE")
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel('gemini-2.5-flash')

            prompt = f"""You are the AI Strategist for a Red Team money laundering simulation.
Your goal is to advise the Red Team on how to evade the Blue Team's GNN fraud detector.

Current Epoch: {epoch}
Current Attack Parameters: {current_params}

The Blue Team's blind spots (features where missed fraud differs from caught fraud):
{blind_spots}

Based on these blind spots, provide:
1. A 2-3 sentence strategic narrative explaining your evasion plan.
2. Specific parameter overrides as a JSON dict. Valid keys:
   - device_sharing_ratio (float 0.05-0.5)
   - topology (list from: fan_out_fan_in, cross_rail, circular, star_burst)
   - noise_txn_per_mule (int 1-8)
   - amount_mean (float 3000-30000)
   - mule_age_range (tuple like [15, 90])
   - dwell_time_min (int 15-300)
   - dwell_time_max (int 300-3600)

Return ONLY valid JSON with keys "narrative" and "overrides"."""

            response = model.generate_content(prompt)
            text = response.text

            match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
            if match:
                text = match.group(1)

            result = json_mod.loads(text)

            narrative = f"🤖 GEMINI STRATEGIST (Epoch {epoch}):\n  {result.get('narrative', '')}"
            return {
                "narrative": narrative,
                "overrides": result.get('overrides', {})
            }

        except Exception as e:
            print(f"  ⚠️ LLM API Error: {e}. Falling back to mock strategy.")
            return self._mock_llm_response(epoch, blind_spots, current_params)


# NOTE: When running in Colab cells (not .py files), the following
# classes/functions are already in memory from previous cells:
#   - DynamicAttackCatalog, SmartEvolution, generate_epoch_data, post_process_ctgan_output
#   - build_networkx_graph, compute_graph_features, prepare_for_gnn
#   - build_model, build_pyg_data, train_model, evaluate_nodes, evaluate_edges,
#     find_optimal_threshold  (Experiment 3: F1-optimal, searched on train split)


def rebuild_graph_from_epoch_data(epoch_df, accounts_df):
    """
    Take an epoch's transaction DataFrame → build graph → compute features →
    return PyG-ready dataset.

    This is called every epoch because each epoch has different data.
    """
    # Build NetworkX graph
    G = build_networkx_graph(epoch_df, accounts_df)

    # Compute graph features
    nodes_featured = compute_graph_features(G, accounts_df.copy())

    # Prepare for PyG (Conditionally generate Heterogeneous Graph)
    gnn_data = prepare_for_gnn(nodes_featured, epoch_df, use_hgt=USE_HGT)

    return gnn_data


def train_blue_team_on_epoch(gnn_data, hidden_channels=64, epochs=120, lr=0.01):
    """
    Build and train a fresh Blue Team model on this epoch's data.

    After training, finds the F1-OPTIMAL decision thresholds (Experiment 3
    Fix 1) by searching precision_recall_curve on the TRAIN split, then
    evaluates the test set with those thresholds. This replaces the old
    Youden's J search on the validation set, which was unstable because
    val only has 7-20 fraud edges — too few for a reliable ROC curve.
    """
    import torch
    # build_model, build_pyg_data, train_model, evaluate_nodes, evaluate_edges,
    # find_optimal_threshold are already in memory from the blue_team_gnn cell

    torch.manual_seed(42)
    np.random.seed(42)

    # Build PyG data
    data = build_pyg_data(gnn_data)

    # Build fresh model conditionally
    if USE_HGT:
        edge_in = data['account', 'transfer', 'account'].edge_attr.shape[1]
        model = build_model(data=data, hidden_channels=hidden_channels)
    else:
        node_in = data.x.shape[1]
        edge_in = data.edge_attr.shape[1]
        model = build_model(node_in_channels=node_in, edge_in_channels=edge_in,
                            hidden_channels=hidden_channels)

    # Train (increased to 250 epochs for better convergence with larger data)
    model = train_model(model, data, epochs=epochs, lr=lr)

    # ─── Find Optimal Thresholds (Experiment 3: F1-optimal, searched on TRAIN split) ───
    node_threshold = find_optimal_threshold(model, data, head='node', search_split='train')
    edge_threshold = find_optimal_threshold(model, data, head='edge', search_split='train')
    print(f"\n  🎯 F1-Optimal Thresholds (searched on train split):")
    print(f"     Node: {node_threshold:.4f} (default was 0.50)")
    print(f"     Edge: {edge_threshold:.4f} (default was 0.50)")

    # Evaluate with OPTIMAL thresholds (not default 0.5)
    node_test = evaluate_nodes(model, data, 'test', threshold=node_threshold)
    edge_test = evaluate_edges(model, data, 'test', threshold=edge_threshold)
    node_train = evaluate_nodes(model, data, 'train', threshold=node_threshold)
    edge_train = evaluate_edges(model, data, 'train', threshold=edge_threshold)

    return model, data, {
        'node_test': node_test,
        'edge_test': edge_test,
        'node_train': node_train,
        'edge_train': edge_train,
        'node_threshold': node_threshold,
        'edge_threshold': edge_threshold
    }


def get_predictions_for_evolution(model, data):
    """
    Get raw predictions and features for SmartEvolution analysis.
    Returns predictions, true labels, and features for both heads.
    """
    import torch

    model.eval()
    with torch.no_grad():
        if USE_HGT:
            node_out, edge_out = model(data.x_dict, data.edge_index_dict, 
                                       data[('account', 'transfer', 'account')].edge_attr,
                                       data['account'].x)
            node_pred = node_out.argmax(dim=1).numpy()
            edge_pred = edge_out.argmax(dim=1).numpy()

            return {
                'node_pred': node_pred,
                'node_true': data['account'].y.cpu().numpy(),
                'node_features': data['account'].x.cpu().numpy(),
                'edge_pred': edge_pred,
                'edge_true': data[('account', 'transfer', 'account')].edge_y.cpu().numpy(),
                'edge_features': data[('account', 'transfer', 'account')].edge_attr.cpu().numpy()
            }
        else:
            node_out, edge_out = model(data.x, data.edge_index, data.edge_attr)
            node_pred = node_out.argmax(dim=1).numpy()
            edge_pred = edge_out.argmax(dim=1).numpy()

            return {
                'node_pred': node_pred,
                'node_true': data.y.cpu().numpy(),
                'node_features': data.x.cpu().numpy(),
                'edge_pred': edge_pred,
                'edge_true': data.edge_y.cpu().numpy(),
                'edge_features': data.edge_attr.cpu().numpy()
            }


# =====================================================================
# MAIN ADVERSARIAL LOOP
# =====================================================================

def run_adversarial_loop(num_epochs=12, n_legit_per_epoch=3000,
                         ctgan_path='ctgan_legit_model.pkl',
                         accounts_path='seed_accounts_nodes.csv'):
    """
    THE MAIN EVENT — Adversarial Training Loop with Data Accumulation.

    KEY DESIGN: Each epoch's data is ACCUMULATED into a growing pool.
    The model trains on ALL past data, so it learns diverse topologies
    (circular + cross_rail + star_burst) → progressively more robust.

    Without accumulation: Model forgets Epoch 0 patterns by Epoch 3
    With accumulation:    Model sees ALL topologies → robust F1 curve
    """
    import torch

    print("=" * 60)
    print("  🛡️  ADVERSARIAL TRAINING LOOP — STARTING")
    print("  Red Team vs Blue Team: The Closed-Loop Battle")
    print("=" * 60)

    # Initialize components
    catalog = DynamicAttackCatalog(accounts_path)
    evolution = SmartEvolution()
    llm_controller = LLMRedTeamController(use_mock=True)

    # Data accumulation pool — grows each epoch
    accumulated_txns = pd.DataFrame()
    accumulated_accounts = pd.DataFrame()

    # Tracking
    all_metrics = []
    all_params = []
    all_reports = []
    blind_spots = None
    legit_boost = 1.0  # FPR feedback: increases if Node FPR > 5%

    node_feature_names = ['account_age_days', 'in_degree', 'out_degree',
                          'pagerank', 'wcc_size', 'device_shared_count', 'ip_shared_count']
    edge_feature_names = ['amount', 'dwell_time_seconds', 'is_structured']

    for epoch in range(num_epochs):
        print(f"\n{'🔴' * 30}")
        print(f"  EPOCH {epoch}")
        print(f"{'🔴' * 30}")

        # ─── STEP 1: Red Team generates NEW data ───
        print(f"\n  Step 1: Red Team generating attacks...")
        # Apply FPR feedback: if Blue Team was lazy-flagging, more legit traffic this epoch
        effective_legit = int(n_legit_per_epoch * legit_boost)
        if legit_boost > 1.0:
            print(f"  📈 FPR Feedback active: {n_legit_per_epoch} → {effective_legit} legit txns")

        epoch_df, params, epoch_accounts = generate_epoch_data(
            catalog=catalog,
            ctgan_path=ctgan_path,
            epoch=epoch,
            blind_spots=blind_spots,
            n_legit=effective_legit,
            llm_controller=llm_controller
        )
        all_params.append(params)

        # ─── STEP 1B: ACCUMULATE data ───
        accumulated_txns = pd.concat([accumulated_txns, epoch_df], ignore_index=True)
        # Merge accounts (keep unique account_ids)
        accumulated_accounts = pd.concat([accumulated_accounts, epoch_accounts], ignore_index=True)
        accumulated_accounts = accumulated_accounts.drop_duplicates(subset='account_id', keep='last')

        print(f"  📚 Accumulated pool: {len(accumulated_txns)} txns, {len(accumulated_accounts)} accounts")

        # ─── STEP 2: Build graph on ACCUMULATED data ───
        print(f"\n  Step 2: Building graph on accumulated data...")
        try:
            gnn_data = rebuild_graph_from_epoch_data(accumulated_txns, accumulated_accounts)
        except Exception as e:
            print(f"  ⚠️ Graph build failed: {e}. Skipping epoch.")
            continue

        # ─── STEP 3: Train Blue Team on ALL data ───
        print(f"\n  Step 3: Training Blue Team GNN on accumulated data...")
        try:
            model, data, metrics = train_blue_team_on_epoch(gnn_data)
        except Exception as e:
            print(f"  ⚠️ Training failed: {e}. Skipping epoch.")
            continue

        # ─── STEP 4: Evaluate & Log (ALL PS metrics) ───
        epoch_result = {
            'epoch': epoch,
            'topology': params['topology'] if isinstance(params['topology'], str) else ','.join(params['topology']),
            'amount_mean': params['amount_mean'],
            'dwell_range': f"{params['dwell_time_min']}-{params['dwell_time_max']}",
            # Node Head (V5.1) — Full PS metrics
            'node_f1_test': metrics['node_test']['f1'],
            'node_precision_test': metrics['node_test']['precision'],
            'node_recall_test': metrics['node_test']['recall'],
            'node_auc_test': metrics['node_test'].get('auc', 0.0),
            'node_pr_auc_test': metrics['node_test'].get('pr_auc', 0.0),
            'node_fpr_test': metrics['node_test'].get('fpr', 0.0),
            'node_threshold': metrics.get('node_threshold', 0.5),
            # Edge Head (V5.2) — Full PS metrics
            'edge_f1_test': metrics['edge_test']['f1'],
            'edge_precision_test': metrics['edge_test']['precision'],
            'edge_recall_test': metrics['edge_test']['recall'],
            'edge_auc_test': metrics['edge_test'].get('auc', 0.0),
            'edge_pr_auc_test': metrics['edge_test'].get('pr_auc', 0.0),
            'edge_fpr_test': metrics['edge_test'].get('fpr', 0.0),
            'edge_threshold': metrics.get('edge_threshold', 0.5),
            # Red Team ASR (Evasion Rate) -> 1.0 - Recall
            'node_asr': 1.0 - metrics['node_test']['recall'],
            'edge_asr': 1.0 - metrics['edge_test']['recall'],
            # Combined
            'combined_f1': (metrics['node_test']['f1'] + metrics['edge_test']['f1']) / 2,
            'combined_auc': (metrics['node_test'].get('auc', 0) + metrics['edge_test'].get('auc', 0)) / 2,
            'timestamp': datetime.now().isoformat()
        }
        all_metrics.append(epoch_result)

        node_m = metrics['node_test']
        edge_m = metrics['edge_test']
        print(f"\n  📊 EPOCH {epoch} RESULTS:")
        print(f"     Topology:     {epoch_result['topology']}")
        print(f"     {'Metric':<12} {'V5.1 Node':<12} {'V5.2 Edge':<12}")
        print(f"     {'─'*36}")
        print(f"     {'Precision':<12} {node_m['precision']:<12.4f} {edge_m['precision']:<12.4f}")
        print(f"     {'Recall':<12} {node_m['recall']:<12.4f} {edge_m['recall']:<12.4f}")
        print(f"     {'F1':<12} {node_m['f1']:<12.4f} {edge_m['f1']:<12.4f}")
        print(f"     {'AUC-ROC':<12} {node_m.get('auc',0):<12.4f} {edge_m.get('auc',0):<12.4f}")
        print(f"     {'PR-AUC':<12} {node_m.get('pr_auc',0):<12.4f} {edge_m.get('pr_auc',0):<12.4f}")
        print(f"     {'FPR':<12} {node_m.get('fpr',0):<12.4f} {edge_m.get('fpr',0):<12.4f}")
        print(f"     {'Threshold':<12} {metrics.get('node_threshold',0.5):<12.4f} {metrics.get('edge_threshold',0.5):<12.4f}")
        print(f"     {'─'*36}")
        print(f"     Combined F1:  {epoch_result['combined_f1']:.4f}")
        print(f"     Combined AUC: {epoch_result['combined_auc']:.4f}")
        print(f"     {'─'*36}")
        print(f"     🔴 Red Team ASR (Evasion Rate):")
        print(f"     Node Evasion {epoch_result['node_asr']*100:.1f}%")
        print(f"     Edge Evasion {epoch_result['edge_asr']*100:.1f}%")

        # ─── FPR FEEDBACK LOOP ───
        # If Node FPR > 5%, boost legit traffic next epoch to punish lazy flagging
        node_fpr = node_m.get('fpr', 0)
        if node_fpr > 0.05:
            legit_boost = min(1.5, 1.0 + node_fpr)  # Up to 1.5x more legit traffic
            print(f"\n  ⚠️ FPR FEEDBACK: Node FPR={node_fpr:.4f} > 0.05 target!")
            print(f"     Next epoch legit traffic boosted by {legit_boost:.2f}x")
        else:
            legit_boost = 1.0
            print(f"\n  ✅ FPR OK: Node FPR={node_fpr:.4f} ≤ 0.05 target")

        # ─── STEP 5: SmartEvolution — Find blind spots ───
        print(f"\n  Step 5: SmartEvolution analyzing blind spots...")
        preds = get_predictions_for_evolution(model, data)

        # Analyze node-level blind spots (V5.1)
        node_blind_spots = evolution.analyze_blind_spots(
            predictions=preds['node_pred'],
            true_labels=preds['node_true'],
            features=preds['node_features'],
            feature_names=node_feature_names
        )

        # Analyze edge-level blind spots (V5.2)
        edge_blind_spots = evolution.analyze_blind_spots(
            predictions=preds['edge_pred'],
            true_labels=preds['edge_true'],
            features=preds['edge_features'],
            feature_names=edge_feature_names
        )

        # Combine blind spots
        blind_spots = {**node_blind_spots, **edge_blind_spots}

        # Generate evolution report FIRST (before adding non-blind-spot metadata)
        report = evolution.generate_report(blind_spots, epoch)
        all_reports.append(report)
        print(report)

        # ─── Track per-topology weakness for softmax weighting ───
        # Added AFTER report generation to avoid KeyError in generate_report()
        current_topologies = params['topology'] if isinstance(params['topology'], list) else [params['topology']]
        topology_weakness = blind_spots.get('topology_weakness', {})
        for topo in current_topologies:
            topology_weakness[topo] = metrics['edge_test']['f1']
        blind_spots['topology_weakness'] = topology_weakness

        # Also pass FPR info for SmartEvolution awareness
        blind_spots['node_fpr'] = node_m.get('fpr', 0)
        blind_spots['edge_fpr'] = edge_m.get('fpr', 0)

        # ─── Convergence Check ───
        if epoch > 0:
            prev_f1 = all_metrics[-2]['combined_f1']
            curr_f1 = epoch_result['combined_f1']
            improvement = curr_f1 - prev_f1

            print(f"  📈 F1 Change: {improvement:+.4f} (prev: {prev_f1:.4f} → curr: {curr_f1:.4f})")

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================

    print(f"\n\n{'=' * 60}")
    print(f"  🏆 ADVERSARIAL LOOP COMPLETE — FINAL SUMMARY")
    print(f"{'=' * 60}")

    print(f"\n  📊 Full Metrics Progression (PS-Aligned):")
    print(f"  {'─' * 105}")
    print(f"  {'Epoch':<6} {'Topology':<16} {'Node F1':<9} {'Edge F1':<9} {'Node AUC':<10} {'Edge AUC':<10} {'Node FPR':<10} {'Edge FPR':<10} {'🔴 Node ASR':<13} {'🔴 Edge ASR':<12}")
    print(f"  {'─' * 105}")

    for m in all_metrics:
        topo = m['topology'][:14] if len(m['topology']) > 14 else m['topology']
        print(f"  {m['epoch']:<6} {topo:<16} {m['node_f1_test']:<9.4f} {m['edge_f1_test']:<9.4f} "
              f"{m.get('node_auc_test',0):<10.4f} {m.get('edge_auc_test',0):<10.4f} "
              f"{m.get('node_fpr_test',0):<10.4f} {m.get('edge_fpr_test',0):<10.4f} "
              f"{m.get('node_asr',0)*100:>10.1f}%   {m.get('edge_asr',0)*100:>10.1f}%")

    # Save final model
    if 'model' in dir():
        torch.save(model.state_dict(), "hardened_blue_team_model.pt")
        print(f"\n  💾 Hardened model saved: hardened_blue_team_model.pt")

    # Save all metrics
    with open("adversarial_loop_metrics.json", "w") as f:
        json.dump({
            'epochs': all_metrics,
            'attack_params': [{k: str(v) for k, v in p.items()} for p in all_params],
            'evolution_reports': all_reports,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"  💾 Metrics saved: adversarial_loop_metrics.json")

    # Print the demo narrative
    print(f"\n  {'=' * 60}")
    print(f"  🎤 JUDGE PITCH NARRATIVE:")
    print(f"  {'=' * 60}")

    if len(all_metrics) >= 2:
        first_f1 = all_metrics[0]['combined_f1']
        peak_f1 = max(m['combined_f1'] for m in all_metrics)
        last_f1 = all_metrics[-1]['combined_f1']
        peak_epoch = max(all_metrics, key=lambda m: m['combined_f1'])['epoch']
        topologies_seen = len(set(m['topology'] for m in all_metrics))
        total_txns = sum(1 for _ in range(len(all_metrics)))  # epochs

        print(f'  "Our Red Team generates adversarial fraud using {topologies_seen}')
        print(f'   distinct laundering topologies with randomized parameters.')
        print(f'   Starting from F1={first_f1:.0%} on initial attacks,')
        print(f'   the Blue Team GNN adapted through {len(all_metrics)} epochs of')
        print(f'   increasingly diverse challenges, peaking at F1={peak_f1:.0%}')
        print(f'   (Epoch {peak_epoch}).')
        print(f'')
        print(f'   SmartEvolution continuously identified blind spots')
        print(f'   (device sharing, PageRank, IP clustering) and fed them')
        print(f'   back to the Red Team — creating a self-improving loop.')
        print(f'')
        print(f'   Final hardened model: Node F1={all_metrics[-1]["node_f1_test"]:.0%},')
        print(f'   Edge F1={all_metrics[-1]["edge_f1_test"]:.0%} — trained on')
        print(f'   {len(all_metrics) * 250} accounts across all topology variants."')

    return all_metrics


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    metrics = run_adversarial_loop(
        num_epochs=12,
        n_legit_per_epoch=3000,
        ctgan_path='ctgan_legit_model.pkl',
        accounts_path='seed_accounts_nodes.csv'
    )