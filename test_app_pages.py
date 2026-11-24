"""
test_app_pages.py
-----------------
Automated test script verifying the data processing, model inference,
graph generation, and report generation backing all 8 pages of the Streamlit application.
"""

import os
import sys
import numpy as np
import pandas as pd

from preprocessing.loader import load_dataset, get_dataset_summary, clean_dataset
from preprocessing.feature_engineering import extract_account_features, get_feature_names
from preprocessing.scaler import scale_features

from models.clustering import compute_elbow_curve, run_kmeans, run_dbscan, run_hierarchical, compute_hierarchical_linkage
from models.anomaly import run_isolation_forest, run_lof
from models.evaluation import evaluate_clusters, compare_all_models

from network.graph_builder import build_transaction_graph, get_graph_summary
from network.tracer import get_account_investigation, trace_transaction_paths, detect_suspicious_network_patterns
from network.network_viz import create_network_plotly, create_ego_network_plotly

from visualization.model_plots import (
    plot_elbow_curve, plot_dendrogram_matplotlib,
    plot_anomaly_score_distribution, plot_clusters_2d_scatter
)
from visualization.dimensionality_reduction import (
    compute_pca, compute_tsne, plot_pca_vs_tsne_side_by_side
)
from reports.reporter import generate_forensic_investigation_report, export_report_to_csv


def test_all_pages():
    print("==================================================")
    print("   FRAUDLINK + NETWORK TRACER - WEB APP TESTS     ")
    print("==================================================\n")
    
    # Page 1 & 2 backend test
    print("[Testing Page 1 & 2: Dataset Loading & Overview]")
    csv_path = "data/paysim_sample.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/transactions.csv"
        
    df_raw = load_dataset(csv_path, sample_size=3000)
    df_clean, audit = clean_dataset(df_raw)
    summary = get_dataset_summary(df_clean)
    assert summary["total_rows"] > 0, "Page 1 summary rows failed"
    print(f"  ✓ Overview & Explorer: {summary['total_rows']} rows, ${summary['total_volume']:,.2f} volume.")
    
    # Page 3 backend test
    print("\n[Testing Page 3: Preprocessing & Scaling]")
    features_df, ground_truth = extract_account_features(df_clean)
    feature_cols = get_feature_names()
    scaled_df, scaler = scale_features(features_df, feature_cols)
    assert scaled_df.shape[1] == 18, f"Expected 18 features, got {scaled_df.shape[1]}"
    print(f"  ✓ Preprocessing: {scaled_df.shape[0]} accounts with {scaled_df.shape[1]} scaled features.")
    
    # Page 4 backend test
    print("\n[Testing Page 4: Machine Learning Analysis]")
    X = scaled_df.values
    ml_results = {
        "K-Means": run_kmeans(X, n_clusters=4),
        "DBSCAN": run_dbscan(X, eps=1.5, min_samples=5),
        "Hierarchical": run_hierarchical(X, n_clusters=4, sample_limit=1000),
        "Isolation Forest": run_isolation_forest(X, contamination=0.03),
        "Local Outlier Factor": run_lof(X, contamination=0.03)
    }
    comparison_table = compare_all_models(X, ml_results, ground_truth=ground_truth)
    assert len(comparison_table) == 5, "Expected 5 models in comparison table"
    print(f"  ✓ ML Analysis: All 5 models evaluated. Flagged outliers: {comparison_table['Flagged Outliers'].tolist()}")
    
    # Page 5 backend test
    print("\n[Testing Page 5: Network Tracer & Graph Intelligence]")
    anomaly_ids = set(scaled_df.index[ml_results["Isolation Forest"]["flags"] == -1])
    anomaly_scores = dict(zip(scaled_df.index, ml_results["Isolation Forest"]["scores"]))
    
    G = build_transaction_graph(df_clean, anomaly_candidates=anomaly_ids, anomaly_scores=anomaly_scores, max_edges=1000)
    g_summary = get_graph_summary(G)
    assert g_summary["num_nodes"] > 0, "Graph failed to build nodes"
    
    test_node = list(anomaly_ids)[0] if anomaly_ids else list(G.nodes())[0]
    inv = get_account_investigation(G, test_node)
    assert inv is not None, "Account investigation failed"
    
    fig_net = create_network_plotly(G, focus_account=test_node, max_nodes=50)
    assert fig_net is not None, "Network Plotly figure failed"
    fig_ego = create_ego_network_plotly(G, center_account=test_node, radius=1)
    assert fig_ego is not None, "Ego Network Plotly figure failed"
    print(f"  ✓ Network Tracer: Graph ({g_summary['num_nodes']} nodes, {g_summary['num_edges']} edges) and Plotly visuals generated.")
    
    # Page 6 backend test
    print("\n[Testing Page 6: Anomaly Detection & Forensic Explainability]")
    report_df = generate_forensic_investigation_report(features_df, ml_results, top_n=50)
    assert not report_df.empty, "Report dataframe should not be empty"
    print(f"  ✓ Anomaly Detection: Generated {len(report_df)} suspect profiles with forensic explanations.")
    
    # Page 7 backend test
    print("\n[Testing Page 7: PCA vs t-SNE]")
    pca_data = compute_pca(X, n_components=2)
    tsne_data = compute_tsne(X, n_components=2, perplexity=25.0, max_samples=800)
    fig_dim = plot_pca_vs_tsne_side_by_side(
        pca_data, tsne_data,
        account_ids=list(scaled_df.index),
        labels=ml_results["K-Means"]["labels"],
        anomaly_flags=ml_results["Isolation Forest"]["flags"]
    )
    assert fig_dim is not None, "PCA vs t-SNE figure failed"
    print(f"  ✓ PCA vs t-SNE: PCA variance {pca_data['total_variance_explained']*100:.1f}%, t-SNE points {len(tsne_data['sample_indices'])}.")
    
    # Page 8 backend test
    print("\n[Testing Page 8: Investigation Reports & CSV Export]")
    csv_str = export_report_to_csv(report_df)
    assert len(csv_str) > 100, "CSV export failed"
    print(f"  ✓ Reports: Successfully exported {len(csv_str)} characters of investigation CSV dossier.")
    
    print("\n==================================================")
    print("   >>> ALL 8 WEB APP MODULE TESTS PASSED! <<<    ")
    print("==================================================")


if __name__ == "__main__":
    test_all_pages()
