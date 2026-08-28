"""
Demo script for Inference: Loading the pre-trained model and detecting fraud in real-time.
This script DOES NOT train the model. It generates a fresh batch of data and evaluates the saved model.
"""
import torch
import pandas as pd
import numpy as np
import logging
import json

import config
from src.blue_team.model import build_model
from src.utils.graph_builder import rebuild_graph_from_epoch_data, build_pyg_data
from src.red_team.data_gen import DynamicAttackCatalog, generate_epoch_data
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_inference_demo():
    logger.info("============================================================")
    logger.info("  🚀 RUNNING INFERENCE DEMO (No Training)")
    logger.info("============================================================")
    
    # 1. Generate a fresh batch of "unseen" data
    logger.info("\n[1/4] Generating fresh 'unseen' transaction data...")
    catalog = DynamicAttackCatalog(config.SEED_ACCOUNTS_PATH)
    
    # Force a hard topology for the demo (e.g., cross_rail and star_burst)
    demo_params = catalog.generate_epoch_params(epoch=99)
    demo_params['topology'] = ['cross_rail', 'star_burst']
    
    epoch_df, params, epoch_accounts = generate_epoch_data(
        catalog=catalog,
        ctgan_path=config.CTGAN_MODEL_PATH,
        epoch=99,
        n_legit=2000, # Smaller batch for fast demo
        llm_controller=None # Disable LLM for faster inference demo
    )
    
    logger.info(f"Generated {len(epoch_df)} transactions and {len(epoch_accounts)} accounts.")
    
    # 2. Build the Graph
    logger.info("\n[2/4] Building Heterogeneous Graph...")
    gnn_data = rebuild_graph_from_epoch_data(epoch_df, epoch_accounts)
    data = build_pyg_data(gnn_data)
    
    # 3. Load Pre-trained Model
    logger.info(f"\n[3/4] Loading hardened model from {config.MODEL_SAVE_PATH}...")
    model = build_model(
        data=data, 
        hidden_channels=config.HIDDEN_CHANNELS,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS
    )
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
        logger.info(f"✅ Successfully loaded model weights.")
    except FileNotFoundError:
        logger.error(f"❌ Model file {config.MODEL_SAVE_PATH} not found. Please run main.py first to train the model.")
        return
        
    # Try to load the optimal thresholds if they were saved in the metrics
    node_threshold = 0.5
    edge_threshold = 0.5
    try:
        with open(config.METRICS_SAVE_PATH, 'r') as f:
            metrics_data = json.load(f)
            last_epoch = metrics_data['epochs'][-1]
            node_threshold = last_epoch.get('node_threshold', 0.5)
            edge_threshold = last_epoch.get('edge_threshold', 0.5)
            logger.info(f"✅ Loaded optimal thresholds from metrics: Node={node_threshold:.4f}, Edge={edge_threshold:.4f}")
    except (FileNotFoundError, KeyError, IndexError):
        logger.info(f"⚠️ Could not load optimal thresholds. Defaulting to 0.5")

    # 4. Run Inference
    logger.info("\n[4/4] Running Inference...")
    model.eval()
    with torch.no_grad():
        node_out, edge_out = model(
            data.x_dict, 
            data.edge_index_dict, 
            data[('account', 'transfer', 'account')].edge_attr,
            data['account'].x
        )
        
        # Get probabilities
        node_probs = torch.softmax(node_out, dim=1)[:, 1].numpy()
        edge_probs = torch.softmax(edge_out, dim=1)[:, 1].numpy()
        
        node_preds = (node_probs >= node_threshold).astype(int)
        edge_preds = (edge_probs >= edge_threshold).astype(int)
        
        node_true = data['account'].y.numpy()
        edge_true = data[('account', 'transfer', 'account')].edge_y.numpy()
        
        logger.info("\n" + "="*60)
        logger.info("  📊 INFERENCE RESULTS")
        logger.info("="*60)
        
        logger.info("\n--- VECTOR 5.1: MULE ACCOUNT DETECTION (Node Head) ---")
        logger.info(classification_report(node_true, node_preds, target_names=['Legitimate', 'Mule'], zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(node_true, node_preds, labels=[0, 1]).ravel()
        logger.info(f"Confusion Matrix: True Positives={tp}, False Positives={fp}")
        logger.info(f"                  False Negatives={fn}, True Negatives={tn}")
        
        logger.info("\n--- VECTOR 5.2: CROSS-RAIL FRAUD DETECTION (Edge Head) ---")
        logger.info(classification_report(edge_true, edge_preds, target_names=['Legitimate', 'Fraud'], zero_division=0))
        
        tn, fp, fn, tp = confusion_matrix(edge_true, edge_preds, labels=[0, 1]).ravel()
        logger.info(f"Confusion Matrix: True Positives={tp}, False Positives={fp}")
        logger.info(f"                  False Negatives={fn}, True Negatives={tn}")
        
        logger.info("\n============================================================")
        logger.info("  ✅ Inference Demo Complete")
        logger.info("============================================================")

if __name__ == "__main__":
    run_inference_demo()
