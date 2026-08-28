"""
MASTERCARD AI DEFENSE LAB — Phase 2: Graph Construction & Feature Engineering

Purpose:
    Converts tabular transaction data into a NetworkX Graph.
    Computes critical structural features for the GNN (Blue Team) to use.
    
Key Graph Features Computed:
    - Node In/Out Degree (captures Fan-in / Fan-out topologies)
    - Node PageRank (finds central aggregator nodes)
    - Weakly Connected Component size (community detection)
    - Device/IP sharing flags (clusters)
    
Output:
    A fully featured dataset ready for PyTorch Geometric (GNN Training).
"""

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

def load_data(txn_path='seed_transactions_edges.csv', nodes_path='seed_accounts_nodes.csv'):
    """Load the generated transaction and account data."""
    print(f"Loading data from {txn_path}...")
    edges_df = pd.read_csv(txn_path)
    nodes_df = pd.read_csv(nodes_path)
    
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    return edges_df, nodes_df

def build_networkx_graph(edges_df, nodes_df):
    """
    Build a Directed Multi-Graph using NetworkX.
    Nodes = Accounts
    Edges = Transactions
    """
    print("Building Directed Multi-Graph...")
    # MultiDiGraph allows multiple transactions between the same two accounts
    G = nx.MultiDiGraph()
    
    # 1. Add Nodes with properties
    print("  -> Adding Nodes...")
    for _, row in nodes_df.iterrows():
        G.add_node(
            row['account_id'],
            account_type=row['account_type'],
            account_age=row['account_age_days'],
            device_hash=row['device_hash'],
            ip_subnet=row['ip_subnet'],
            is_mule=row['is_mule']
        )
        
    # 2. Add Edges with properties
    print("  -> Adding Edges...")
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['sender'],
            row['receiver'],
            txn_id=row['txn_id'],
            amount=row['amount'],
            rail=row['rail'],
            dwell_time=row['dwell_time_seconds'],
            is_structured=row['is_structured'],
            shared_hash=row['shared_hash'],
            is_fraud=row['is_fraud'],
            timestamp=row['timestamp']
        )
        
    print(f"✅ Graph built: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
    return G

def compute_graph_features(G, nodes_df):
    """
    Compute structural features for each node and append to DataFrame.
    These features are CRITICAL for the GNN to detect mule networks.
    """
    print("\nComputing Graph-Level Features (This may take a moment)...")
    
    # Ensure graph acts as a simple DiGraph for structural metrics
    G_simple = nx.DiGraph(G)
    
    # 1. Degree Centrality (Fan-in / Fan-out)
    print("  -> Computing In/Out Degrees...")
    in_degrees = dict(G_simple.in_degree())
    out_degrees = dict(G_simple.out_degree())
    
    # 2. PageRank (Finds central hubs / aggregators)
    print("  -> Computing PageRank...")
    pagerank = nx.pagerank(G_simple, weight='amount')
    
    # 3. Community Detection / Clustering (Approximation via weak components)
    print("  -> Computing Connectivity components...")
    wcc = list(nx.weakly_connected_components(G_simple))
    wcc_size = {node: len(comp) for comp in wcc for node in comp}

    # Map features back to the Node DataFrame
    nodes_df['in_degree'] = nodes_df['account_id'].map(in_degrees).fillna(0)
    nodes_df['out_degree'] = nodes_df['account_id'].map(out_degrees).fillna(0)
    nodes_df['pagerank'] = nodes_df['account_id'].map(pagerank).fillna(0)
    nodes_df['wcc_size'] = nodes_df['account_id'].map(wcc_size).fillna(1)
    
    # 4. Compute 'device_shared_count'
    print("  -> Computing Device & IP sharing signals...")
    device_counts = nodes_df['device_hash'].value_counts().to_dict()
    nodes_df['device_shared_count'] = nodes_df['device_hash'].map(device_counts)
    
    ip_counts = nodes_df['ip_subnet'].value_counts().to_dict()
    nodes_df['ip_shared_count'] = nodes_df['ip_subnet'].map(ip_counts)

    return nodes_df

def prepare_for_gnn(nodes_df, edges_df, use_hgt=False):
    """
    Prepare the final feature matrices required by PyTorch Geometric (PyG).
    PyG needs:
    1. Node Feature Matrix (X)
    2. Edge Index (adjacency list)
    3. Edge Features
    4. Labels (Y)
    """
    print(f"\nFormatting data for PyTorch Geometric (GNN, Heterogeneous={use_hgt})...")
    
    # Map account UUIDs to integer IDs (0 to N-1) for PyG compatibility
    node_mapping = {acc: i for i, acc in enumerate(nodes_df['account_id'].unique())}
    
    # 1. Edge Index
    source_nodes = edges_df['sender'].map(node_mapping).values
    target_nodes = edges_df['receiver'].map(node_mapping).values
    edge_index = np.vstack((source_nodes, target_nodes))
    
    # 2. Node Features (Normalize numeric columns)
    feature_cols = ['account_age_days', 'in_degree', 'out_degree', 'pagerank', 
                    'wcc_size', 'device_shared_count', 'ip_shared_count']
    
    # Simple Min-Max scaling for node features
    X_nodes = nodes_df[feature_cols].copy()
    for col in feature_cols:
        X_nodes[col] = (X_nodes[col] - X_nodes[col].min()) / (X_nodes[col].max() - X_nodes[col].min() + 1e-9)
        
    node_features = X_nodes.values
    node_labels = nodes_df['is_mule'].values
    
    # 3. Edge Features
    edge_cols = ['amount', 'dwell_time_seconds', 'is_structured']
    X_edges = edges_df[edge_cols].copy()
    # Velocity feature: catches fast-moving money (mules move funds quickly)
    X_edges['velocity'] = edges_df['amount'] / (edges_df['dwell_time_seconds'] + 1)
    for col in ['amount', 'dwell_time_seconds', 'velocity']:
        X_edges[col] = (X_edges[col] - X_edges[col].min()) / (X_edges[col].max() - X_edges[col].min() + 1e-9)
    
    edge_features = X_edges.values
    edge_labels = edges_df['is_fraud'].values

    if use_hgt:
        # Create Heterogeneous Graph structure
        # Map device and ip string IDs to integer indices
        device_mapping = {dev: i for i, dev in enumerate(nodes_df['device_hash'].unique())}
        ip_mapping = {ip: i for i, ip in enumerate(nodes_df['ip_subnet'].unique())}
        
        # Create 'uses' edges for Device and IP
        acc_nodes_for_device = nodes_df['account_id'].map(node_mapping).values
        dev_nodes = nodes_df['device_hash'].map(device_mapping).values
        edge_index_uses_device = np.vstack((acc_nodes_for_device, dev_nodes))
        
        acc_nodes_for_ip = nodes_df['account_id'].map(node_mapping).values
        ip_nodes = nodes_df['ip_subnet'].map(ip_mapping).values
        edge_index_uses_ip = np.vstack((acc_nodes_for_ip, ip_nodes))
        
        # Package into a dictionary structured for HeteroData conversion
        graph_data = {
            'heterogeneous': True,
            'node_features_account': node_features,
            'node_labels_account': node_labels,
            'num_nodes_device': len(device_mapping),
            'num_nodes_ip': len(ip_mapping),
            'edge_index_transfer': edge_index,
            'edge_features_transfer': edge_features,
            'edge_labels_transfer': edge_labels,
            'edge_index_uses_device': edge_index_uses_device,
            'edge_index_uses_ip': edge_index_uses_ip,
            'node_mapping': node_mapping
        }
    else:
        # Package into a dictionary
        graph_data = {
            'heterogeneous': False,
            'edge_index': edge_index,
            'node_features': node_features,
            'node_labels': node_labels,
            'edge_features': edge_features,
            'edge_labels': edge_labels,
            'node_mapping': node_mapping # Save to reverse map later
        }
    
    print("✅ PyTorch Geometric data package created.")
    return graph_data

if __name__ == "__main__":
    print("=" * 60)
    print("  MASTERCARD AI DEFENSE LAB — GRAPH CONSTRUCTION")
    print("  Phase 2: NetworkX to PyTorch Geometric")
    print("=" * 60)
    
    # 1. Load Data
    edges_df, nodes_df = load_data()
    
    # 2. Build NetworkX Graph
    G = build_networkx_graph(edges_df, nodes_df)
    
    # 3. Compute Graph Features (PageRank, Degrees)
    nodes_featured = compute_graph_features(G, nodes_df)
    
    # Quick sanity check
    print("\n📊 Feature Sanity Check (Average values):")
    print("Legitimate Accounts:")
    print(nodes_featured[nodes_featured['is_mule']==0][['in_degree', 'out_degree', 'device_shared_count']].mean().round(2))
    print("\nMule Accounts:")
    print(nodes_featured[nodes_featured['is_mule']==1][['in_degree', 'out_degree', 'device_shared_count']].mean().round(2))
    
    # 4. Prepare for GNN
    gnn_dataset = prepare_for_gnn(nodes_featured, edges_df)
    
    # 5. Save everything
    print("\n💾 Saving processed Graph Data...")
    
    # Save enriched node features
    nodes_featured.to_csv("graph_nodes_featured.csv", index=False)
    
    # Save PyG dict
    with open("gnn_dataset.pkl", "wb") as f:
        pickle.dump(gnn_dataset, f)
        
    print("✅ Saved: graph_nodes_featured.csv")
    print("✅ Saved: gnn_dataset.pkl")
    print("\nNext Step: Phase 3 (Blue Team v1 - Train GraphSAGE/GAT model on gnn_dataset.pkl)")
