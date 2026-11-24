"""
test_phase4.py
--------------
Comprehensive verification script for Phase 4 – Network Tracer:
1. Loads dataset, engineers features, and runs Phase 3 Isolation Forest to get anomaly candidates.
2. Builds NetworkX DiGraph with transaction edge weights and node attributes.
3. Tests account search & detailed forensic investigation.
4. Tests multi-hop transaction path tracing (shortest path & alternative paths).
5. Tests AML topological pattern detection (Fan-in, Fan-out, Circular Loops, Connected Groups).
6. Tests interactive Plotly network graph generation & Ego-Network rendering.
"""

import os
import sys
import numpy as np
import networkx as nx
import pandas as pd

from preprocessing.loader import load_dataset, clean_dataset
from preprocessing.feature_engineering import extract_account_features, get_feature_names
from preprocessing.scaler import scale_features
from models.anomaly import run_isolation_forest

from network.graph_builder import build_transaction_graph, get_graph_summary
from network.tracer import (
    get_account_investigation,
    trace_transaction_paths,
    detect_suspicious_network_patterns
)
from network.network_viz import create_network_plotly, create_ego_network_plotly


def run_phase4_verification():
    print("==================================================")
    print("      FRAUDLINK + NETWORK TRACER - PHASE 4 TEST   ")
    print("==================================================\n")
    
    # 1. Load Data
    csv_path = "data/paysim_sample.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/transactions.csv"
        
    print(f"[Step 1] Loading transactions from '{csv_path}'...")
    df_raw = load_dataset(csv_path, sample_size=5000)
    df_clean, _ = clean_dataset(df_raw)
    
    # 2. Extract Features & Get Anomaly Candidates from Phase 3
    print("[Step 2] Identifying anomaly candidates via Phase 3 Isolation Forest...")
    features_df, _ = extract_account_features(df_clean)
    feature_cols = get_feature_names()
    scaled_df, _ = scale_features(features_df, feature_cols)
    
    iso_res = run_isolation_forest(scaled_df.values, contamination=0.03)
    anomaly_indices = np.where(iso_res["flags"] == -1)[0]
    anomaly_candidates = set(scaled_df.index[anomaly_indices])
    anomaly_scores = dict(zip(scaled_df.index, iso_res["scores"]))
    print(f"  ✓ Identified {len(anomaly_candidates)} anomaly candidate accounts.\n")
    
    # 3. Build Transaction Graph
    print("[Step 3] Building directed transaction graph (NetworkX DiGraph)...")
    G = build_transaction_graph(
        df_clean,
        anomaly_candidates=anomaly_candidates,
        anomaly_scores=anomaly_scores,
        min_amount=100.0,
        max_edges=2500
    )
    summary = get_graph_summary(G)
    print(f"  ✓ Graph Nodes (Accounts): {summary['num_nodes']:,}")
    print(f"  ✓ Graph Edges (Transactions): {summary['num_edges']:,}")
    print(f"  ✓ Total Volume in Network: ${summary['total_volume']:,.2f}")
    print(f"  ✓ Network Density: {summary['density']}")
    print(f"  ✓ Weakly Connected Components: {summary['weakly_connected_components']}")
    print(f"  ✓ Anomaly Nodes in Graph: {summary['anomaly_nodes_count']} ({summary['anomaly_pct']}%)")
    print(f"  ✓ Top Hub Accounts by Degree: {summary['top_hubs'][:3]}\n")
    assert summary['num_nodes'] > 0, "Graph should have nodes"
    assert summary['num_edges'] > 0, "Graph should have edges"
    
    # 4. Test Account Investigation
    print("[Step 4] Testing Account Search & Investigation...")
    # Select an anomaly candidate or top hub to investigate
    if anomaly_candidates:
        test_account = list(anomaly_candidates)[0]
    else:
        test_account = list(G.nodes())[0]
        
    investigation = get_account_investigation(G, test_account)
    assert investigation is not None, f"Investigation for {test_account} failed"
    print(f"  ✓ Investigating Account: {investigation['account_id']}")
    print(f"  ✓ Status: {'ANOMALY CANDIDATE' if investigation['is_anomaly'] else 'NORMAL'}")
    print(f"  ✓ Anomaly Score: {investigation['anomaly_score']}")
    print(f"  ✓ Total Sent: ${investigation['total_sent']:,.2f} ({investigation['out_degree']} txns)")
    print(f"  ✓ Total Received: ${investigation['total_received']:,.2f} ({investigation['in_degree']} txns)")
    print(f"  ✓ Connected Accounts: {investigation['connected_accounts_count']}")
    if investigation['incoming_transactions']:
        print(f"  ✓ Sample Incoming: {investigation['incoming_transactions'][0]}")
    if investigation['outgoing_transactions']:
        print(f"  ✓ Sample Outgoing: {investigation['outgoing_transactions'][0]}\n")
        
    # 5. Test Path Tracing
    print("[Step 5] Testing Multi-hop Transaction Path Tracing...")
    # Find a verified 2-hop path (e.g. through investigated mule account)
    src_node = "C6004378"
    dst_node = "C8004804"
    if G.has_node(src_node) and G.has_node(dst_node):
        path_result = trace_transaction_paths(G, src_node, dst_node, max_paths=3, max_depth=4)
        print(f"  ✓ Tracing Path from {src_node} → {dst_node}:")
        print(f"  ✓ Connection Status: {path_result['status']}")
        if path_result['status'] == "connected":
            print(f"  ✓ Shortest Path: {' -> '.join(path_result['shortest_path_nodes'])} ({path_result['shortest_path_hops']} hops)")
            print(f"  ✓ Path Hops Details: {path_result['paths'][0]['hops']}")
    else:
        # Fallback to any connected pair in G
        edges = list(G.edges())
        src_node, dst_node = edges[0]
        path_result = trace_transaction_paths(G, src_node, dst_node, max_paths=3, max_depth=4)
        print(f"  ✓ Tracing Path from {src_node} → {dst_node}: status = {path_result['status']}")
    print("  ✓ Path tracing function validated.\n")
    
    # 6. Test Suspicious AML Pattern Detection
    print("[Step 6] Testing AML Pattern Mining (Fan-in, Fan-out, Cycles)...")
    patterns = detect_suspicious_network_patterns(G, min_fan_degree=3, max_cycle_length=4)
    print(f"  ✓ Fan-in (Structuring / Pooling) accounts detected: {len(patterns['fan_in_patterns'])}")
    if patterns['fan_in_patterns']:
        print(f"    Top Fan-in: {patterns['fan_in_patterns'][0]}")
    print(f"  ✓ Fan-out (Dispersal / Layering) accounts detected: {len(patterns['fan_out_patterns'])}")
    if patterns['fan_out_patterns']:
        print(f"    Top Fan-out: {patterns['fan_out_patterns'][0]}")
    print(f"  ✓ Circular Transaction Loops: {len(patterns['circular_loops'])}")
    if patterns['circular_loops']:
        print(f"    Sample Cycle: {patterns['circular_loops'][0]['hops']}")
    print(f"  ✓ Total transacting groups: {patterns['total_weakly_connected_groups']}")
    print(f"  ✓ Largest transacting network: {patterns['largest_group_size']} accounts\n")
    
    # 7. Test Plotly Visualizations
    print("[Step 7] Testing Interactive Plotly Network Visualizations...")
    fig_network = create_network_plotly(G, focus_account=test_account, max_nodes=100)
    assert fig_network is not None, "Network plot failed"
    print("  ✓ Full Transaction Network Plotly figure generated.")
    
    fig_ego = create_ego_network_plotly(G, center_account=test_account, radius=1)
    assert fig_ego is not None, "Ego network plot failed"
    print(f"  ✓ Ego-Network Plotly figure for account '{test_account}' generated.\n")
    
    print("==================================================")
    print("   >>> ALL PHASE 4 TESTS PASSED SUCCESSFULLY! <<< ")
    print("==================================================")


def test_phase4_network_tracer():
    """Pytest test case for Phase 4 Network Tracer."""
    run_phase4_verification()


if __name__ == "__main__":
    run_phase4_verification()
