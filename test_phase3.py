"""
test_phase3.py
--------------
Comprehensive verification script for Phase 3:
1. Loads dataset and extracts account features.
2. Standardizes features with StandardScaler.
3. Tests K-Means Clustering & Elbow method.
4. Tests DBSCAN (Core, Border, Noise detection).
5. Tests Hierarchical Clustering & Linkage Matrix.
6. Tests Isolation Forest (Anomaly scoring).
7. Tests Local Outlier Factor (LOF).
8. Tests Model Evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz).
9. Tests Master Comparison Table generation.
10. Tests Visualization figure generation.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from preprocessing.loader import load_dataset, clean_dataset
from preprocessing.feature_engineering import extract_account_features, get_feature_names
from preprocessing.scaler import scale_features

from models.clustering import compute_elbow_curve, run_kmeans, run_dbscan, run_hierarchical, compute_hierarchical_linkage
from models.anomaly import run_isolation_forest, run_lof
from models.evaluation import evaluate_clusters, compare_all_models
from visualization.model_plots import (
    plot_elbow_curve,
    plot_dendrogram_matplotlib,
    plot_anomaly_score_distribution,
    plot_clusters_2d_scatter
)


def run_phase3_verification():
    print("==================================================")
    print("      FRAUDLINK + NETWORK TRACER - PHASE 3 TEST   ")
    print("==================================================\n")
    
    # 1. Load Data & Prepare Scaled Account Profiles
    csv_path = "data/paysim_sample.csv"
    if not os.path.exists(csv_path):
        csv_path = "data/transactions.csv"
        
    print(f"[Step 1] Loading data from '{csv_path}'...")
    df_raw = load_dataset(csv_path, sample_size=5000)
    df_clean, _ = clean_dataset(df_raw)
    
    print("[Step 2] Engineering account-level features...")
    features_df, ground_truth = extract_account_features(df_clean)
    feature_cols = get_feature_names()
    scaled_df, scaler = scale_features(features_df, feature_cols)
    X = scaled_df.values
    account_ids = list(scaled_df.index)
    print(f"  ✓ Prepared feature matrix: {X.shape[0]} accounts, {X.shape[1]} features.\n")
    
    model_results = {}
    
    # 2. Test K-Means & Elbow Curve
    print("[Step 3] Testing K-Means Clustering & Elbow Analysis...")
    elbow_data = compute_elbow_curve(X, k_range=range(2, 6))
    print(f"  ✓ Elbow inertias for K=2..5: {[round(i, 1) for i in elbow_data['inertias']]}")
    print(f"  ✓ Silhouette scores for K=2..5: {[round(s, 3) for s in elbow_data['silhouettes']]}")
    
    km_res = run_kmeans(X, n_clusters=4)
    model_results["K-Means (k=4)"] = km_res
    print(f"  ✓ K-Means fitted: Inertia = {km_res['inertia']:.2f}")
    print(f"  ✓ Cluster breakdown: {km_res['cluster_counts']}")
    print(f"  ✓ Minority cluster outliers: {np.sum(km_res['anomaly_flags'] == -1)}\n")
    
    # 3. Test DBSCAN
    print("[Step 4] Testing DBSCAN Density Clustering...")
    db_res = run_dbscan(X, eps=1.5, min_samples=5)
    model_results["DBSCAN"] = db_res
    print(f"  ✓ DBSCAN clusters discovered: {db_res['num_clusters']}")
    print(f"  ✓ Noise points (Anomalies): {db_res['noise_count']} ({db_res['noise_pct']}%)")
    point_types = pd.Series(db_res['point_types']).value_counts().to_dict()
    print(f"  ✓ Point classification: {point_types}\n")
    
    # 4. Test Hierarchical Clustering & Linkage Matrix
    print("[Step 5] Testing Agglomerative Hierarchical Clustering...")
    hier_res = run_hierarchical(X, n_clusters=4, linkage_method="ward", sample_limit=1500)
    model_results["Hierarchical"] = hier_res
    print(f"  ✓ Hierarchical clusters: {hier_res['num_clusters']} (linkage: {hier_res['linkage']})")
    print(f"  ✓ Cluster breakdown: {hier_res['cluster_counts']}")
    linkage_mat = compute_hierarchical_linkage(X, method="ward", sample_size=100)
    print(f"  ✓ Computed linkage matrix shape: {linkage_mat.shape}\n")
    
    # 5. Test Isolation Forest
    print("[Step 6] Testing Isolation Forest...")
    iso_res = run_isolation_forest(X, contamination=0.03)
    model_results["Isolation Forest"] = iso_res
    print(f"  ✓ Isolation Forest outliers flagged: {iso_res['outlier_count']} ({iso_res['outlier_pct']}%)")
    print(f"  ✓ Score range: min={iso_res['scores'].min():.3f}, max={iso_res['scores'].max():.3f}, mean={iso_res['scores'].mean():.3f}\n")
    
    # 6. Test Local Outlier Factor
    print("[Step 7] Testing Local Outlier Factor (LOF)...")
    lof_res = run_lof(X, n_neighbors=20, contamination=0.03)
    model_results["Local Outlier Factor"] = lof_res
    print(f"  ✓ LOF outliers flagged: {lof_res['outlier_count']} ({lof_res['outlier_pct']}%)")
    print(f"  ✓ Score range: min={lof_res['scores'].min():.3f}, max={lof_res['scores'].max():.3f}, mean={lof_res['scores'].mean():.3f}\n")
    
    # 7. Test Evaluation & Model Comparison
    print("[Step 8] Benchmarking & Model Comparison Table...")
    comparison_df = compare_all_models(X, model_results, ground_truth=ground_truth)
    print("\n--- MASTER ALGORITHM COMPARISON ---")
    print(comparison_df.to_string(index=False))
    print("-----------------------------------\n")
    
    # 8. Test Visualizations
    print("[Step 9] Generating Model Visualizations...")
    fig_elbow = plot_elbow_curve(elbow_data)
    assert fig_elbow is not None, "Elbow plot failed"
    print("  ✓ Elbow Curve Plotly figure generated.")
    
    fig_dendro = plot_dendrogram_matplotlib(linkage_mat, max_d=25.0)
    assert fig_dendro is not None, "Dendrogram failed"
    print("  ✓ Dendrogram Matplotlib figure generated.")
    
    fig_iso = plot_anomaly_score_distribution(iso_res["scores"], iso_res["flags"], title="Isolation Forest Scores")
    assert fig_iso is not None, "Score distribution plot failed"
    print("  ✓ Anomaly Score Distribution Plotly figure generated.")
    
    # 2D PCA for cluster scatter test
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X)
    fig_scatter = plot_clusters_2d_scatter(coords_2d, km_res["labels"], account_ids, km_res["anomaly_flags"])
    assert fig_scatter is not None, "2D cluster scatter plot failed"
    print("  ✓ 2D Cluster Scatter Plotly figure generated.\n")
    
    print("==================================================")
    print("   >>> ALL PHASE 3 TESTS PASSED SUCCESSFULLY! <<< ")
    print("==================================================")


def test_phase3_ml_pipeline():
    """Pytest test case for Phase 3 Unsupervised ML models."""
    run_phase3_verification()


if __name__ == "__main__":
    run_phase3_verification()
