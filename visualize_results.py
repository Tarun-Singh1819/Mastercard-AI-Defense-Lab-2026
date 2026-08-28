"""
MASTERCARD AI DEFENSE LAB — Phase 5: Presentation & Visualization
Generates high-impact charts for the Hackathon Pitch.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_metrics(filepath='adversarial_loop_metrics.json'):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data['epochs']
    except FileNotFoundError:
        print(f"Error: {filepath} not found. Run the adversarial loop first.")
        return None

def plot_cat_and_mouse(epochs):
    """
    Dual-Axis Chart: Blue Team F1 (rising) vs Red Team ASR (falling).
    This tells the main story of the closed-loop system.
    """
    x = [m['epoch'] for m in epochs]
    blue_f1 = [m['combined_f1'] * 100 for m in epochs]
    
    # Calculate combined ASR (average of node and edge ASR)
    red_asr = [((m.get('node_asr', 0) + m.get('edge_asr', 0)) / 2) * 100 for m in epochs]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis 1: Blue Team F1
    color = 'tab:blue'
    ax1.set_xlabel('Adversarial Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Blue Team Combined F1 (%)', color=color, fontsize=12, fontweight='bold')
    ax1.plot(x, blue_f1, color=color, marker='o', linewidth=3, markersize=8, label='Blue Team F1')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(40, 100)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Axis 2: Red Team ASR
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Red Team Attack Success Rate (%)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(x, red_asr, color=color, marker='X', linewidth=3, markersize=8, linestyle='--', label='Red Team ASR')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 60)

    # Title & Save
    plt.title('The "Cat & Mouse" Effect: Self-Improving Adversarial System', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    plt.savefig('pitch_cat_and_mouse.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: pitch_cat_and_mouse.png")
    plt.close()

def plot_fpr_guardrails(epochs):
    """
    Shows that FPR stayed strictly below the 5% industry threshold.
    """
    x = [m['epoch'] for m in epochs]
    node_fpr = [m.get('node_fpr_test', 0) * 100 for m in epochs]
    edge_fpr = [m.get('edge_fpr_test', 0) * 100 for m in epochs]

    plt.figure(figsize=(10, 6))
    
    plt.plot(x, node_fpr, color='#4B0082', marker='o', linewidth=2, label='Node FPR (Account Blocking)')
    plt.plot(x, edge_fpr, color='#006400', marker='s', linewidth=2, label='Edge FPR (Txn Declines)')
    
    # The 5% Hard Limit Line
    plt.axhline(y=5.0, color='red', linestyle='-', linewidth=2, label='Industry Max Tolerable FPR (5%)')
    plt.fill_between(x, 5.0, 10.0, color='red', alpha=0.1)

    plt.title('FPR Guardrails: Maintaining Live-Payment Viability', fontsize=14, fontweight='bold')
    plt.xlabel('Adversarial Epochs', fontsize=12)
    plt.ylabel('False Positive Rate (%)', fontsize=12)
    plt.ylim(0, 10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='upper right')
    
    plt.savefig('pitch_fpr_guardrails.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: pitch_fpr_guardrails.png")
    plt.close()

if __name__ == "__main__":
    print("="*50)
    print(" GENERATING PITCH VISUALIZATIONS")
    print("="*50)
    
    epochs = load_metrics()
    if epochs:
        plot_cat_and_mouse(epochs)
        plot_fpr_guardrails(epochs)
        print("\nAll visualizations ready for the presentation deck!")
