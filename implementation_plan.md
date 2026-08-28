# 🛡️ Mastercard AI Defense Lab — Implementation Plan

## ✅ What's Already Done

| # | Script | Status | Output Files |
|---|--------|--------|-------------|
| 1 | `generate_seed_data.py` | ✅ Run & Saved | `seed_accounts_nodes.csv`, `seed_transactions_edges.csv` |
| 2 | `train_ctgan.py` | ✅ Run & Saved | `ctgan_legit_model.pkl`, `ctgan_fraud_model.pkl` |
| 3 | `build_graph.py` | 📝 Written, Not Run Yet | Will create `gnn_dataset.pkl` |

---

## 🔧 Changes Needed in Existing Scripts

### Change 1: `train_ctgan.py` — Drop CTGAN_FRAUD, Add Post-Processing

**Why:** CTGAN_FRAUD trained on only 190 samples → mode collapse → generates ₹1.18L amounts instead of structured ₹9K. Fraud features will come from `DynamicAttackCatalog` instead.

**What to change:**
- Remove CTGAN_FRAUD training entirely (lines ~250-270)
- Add `post_process_ctgan_output()` function that clips negative amounts (min ₹50) and negative dwell times (min 10s) after sampling
- Only save `ctgan_legit_model.pkl`

> [!NOTE]
> **You already ran this script.** No need to re-run now. The `ctgan_legit_model.pkl` is usable. We'll apply post-processing when we SAMPLE from it later in the adversarial loop. The `ctgan_fraud_model.pkl` file can be ignored.

### Change 2: `build_graph.py` — No Changes Needed ✅

This script is ready to run as-is. It loads seed CSVs, builds NetworkX graph, computes features (PageRank, degree, device sharing), and outputs `gnn_dataset.pkl`.

---

## 🆕 New Scripts to Build (In Order)

### Script 4: `blue_team_gnn.py` (Phase 3)
**What:** Train a GraphSAGE + GAT model (the Blue Team detector) on `gnn_dataset.pkl`
**Input:** `gnn_dataset.pkl` (from `build_graph.py`)
**Output:** `blue_team_model.pt` (trained PyTorch model), evaluation metrics (Precision, Recall, F1, AUC)
**Key decisions:**
- 2-layer GraphSAGE (prevents over-smoothing)
- Node classification: `is_mule = 0 or 1`
- Focal Loss (handles class imbalance — only 3.5% fraud)
- Train/Val/Test split: 60/20/20

---

### Script 5: `dynamic_attack_catalog.py` (Phase 4A — Red Team Engine)
**What:** The `DynamicAttackCatalog` class that generates fraud data with RANDOMIZED parameters per epoch
**Input:** `seed_accounts_nodes.csv` (account metadata), epoch number, blind_spots (from previous epoch)
**Output:** Fraud transaction DataFrame with randomized features but consistent graph topology
**Key design:**
- Parameter Fuzzing: structuring threshold, fan-out degree, dwell time — all randomized per epoch
- Domain Randomization: amounts change every epoch → GNN forced to learn TOPOLOGY, not VALUES
- Topology mixing: Fan-Out/Fan-In, Circular, Cross-Rail randomly selected
- Noise injection: 5-40% legitimate transactions mixed into mule accounts
- SmartEvolution integration: blind spots shift parameters toward Blue Team weaknesses

---

### Script 6: `adversarial_loop.py` (Phase 4B — The Main Event)
**What:** The complete closed-loop training engine that ties everything together
**Input:** All models and data from previous scripts
**Output:** Epoch-by-epoch metrics, evolution reports, final hardened Blue Team model
**Flow per epoch:**
1. `DynamicAttackCatalog` generates fraud data (randomized params)
2. `CTGAN_LEGIT` generates legitimate background traffic (post-processed)
3. Combine → Fidelity Validator (quality gate)
4. Build graph → Blue Team GNN trains/fine-tunes
5. Evaluate → SmartEvolution analyzes blind spots
6. Feed blind spots back to Red Team → Repeat

---

### Script 7: `visualize_results.py` (Phase 5 — Demo)
**What:** Generate the visual demo for judges
**Output:**
- F1 improvement chart across epochs (bar chart)
- Cross-rail graph visualization (money path highlighted)
- Evolution report (what Red Team learned each epoch)
- Confusion matrix per epoch

---

## 📋 Execution Order (Checklist)

```
Phase 1 (Data Engine) — DONE ✅
  [x] 1. Run generate_seed_data.py        → seed CSVs
  [x] 2. Run train_ctgan.py               → ctgan_legit_model.pkl

Phase 2 (Graph Construction) — DO NOW
  [ ] 3. Run build_graph.py               → gnn_dataset.pkl

Phase 3 (Blue Team v1) — NEXT
  [ ] 4. Build & Run blue_team_gnn.py     → blue_team_model.pt + baseline metrics

Phase 4 (Adversarial Loop) — CORE
  [ ] 5. Build dynamic_attack_catalog.py   → DynamicAttackCatalog class
  [ ] 6. Build & Run adversarial_loop.py   → epoch metrics + hardened model

Phase 5 (Demo & Viz) — FINAL
  [ ] 7. Build & Run visualize_results.py  → charts, graphs, demo materials
```

## 🗺️ File Dependency Map

```
generate_seed_data.py
    ├── seed_accounts_nodes.csv ──────────────┐
    └── seed_transactions_edges.csv ──────┐   │
                                          │   │
train_ctgan.py                            │   │
    └── ctgan_legit_model.pkl ────────┐   │   │
                                      │   │   │
build_graph.py  ◄─────────────────────│───┘   │
    └── gnn_dataset.pkl ──────────┐   │       │
                                  │   │       │
blue_team_gnn.py ◄────────────────┘   │       │
    └── blue_team_model.pt ───┐       │       │
                              │       │       │
dynamic_attack_catalog.py ◄───│───────│───────┘
    └── (generates data on-the-fly per epoch)
                              │       │
adversarial_loop.py ◄─────────┴───────┘
    ├── Uses: blue_team_model.pt (fine-tunes)
    ├── Uses: ctgan_legit_model.pkl (legit traffic)
    ├── Uses: DynamicAttackCatalog (fraud generation)
    ├── Uses: FidelityValidator (quality gate)
    ├── Uses: SmartEvolution (blind spot analysis)
    └── Output: epoch_metrics.json, hardened_model.pt

visualize_results.py ◄── epoch_metrics.json
    └── Output: charts, graphs, demo materials
```
