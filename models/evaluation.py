"""
models/evaluation.py
--------------------
Evaluation and benchmarking module for unsupervised clustering and anomaly models.

Metrics Computed:
1. Silhouette Score:
   - Range: [-1, +1]. Higher is better.
   - Measures how well-separated and cohesive clusters are.
2. Davies-Bouldin Index (DBI):
   - Range: [0, +inf]. Lower is better.
   - Ratio of within-cluster distance to between-cluster distance.
3. Calinski-Harabasz Index (Variance Ratio Criterion):
   - Range: [0, +inf]. Higher is better.
   - Ratio of between-cluster dispersion to within-cluster dispersion.
4. Outlier Count & Proportion (%):
   - Compares how many accounts each algorithm flagged as suspicious.
5. Ground Truth Alignment (Optional):
   - Precision, Recall, and F1 if ground-truth labels exist (for academic comparison only).
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str = "Clustering Model"
) -> Dict[str, Any]:
    """
    Compute intrinsic unsupervised clustering metrics.
    
    Filters out noise points (label = -1) for density algorithms like DBSCAN
    so that geometric metrics reflect valid cluster quality.
    """
    unique_labels = set(labels) - {-1}
    num_clusters = len(unique_labels)
    total_samples = len(X)
    
    # If fewer than 2 valid clusters exist, cluster quality metrics are undefined
    if num_clusters < 2:
        return {
            "model_name": model_name,
            "num_clusters": num_clusters,
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "valid_evaluation": False,
            "note": "Requires at least 2 distinct clusters for evaluation."
        }
        
    # Mask out noise points for geometric metrics
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    # Fast silhouette calculation with subsampling if N > 2500
    if len(X_valid) > 2500:
        sample_size = 2000
    else:
        sample_size = None
        
    sil = float(silhouette_score(X_valid, labels_valid, sample_size=sample_size))
    dbi = float(davies_bouldin_score(X_valid, labels_valid))
    chi = float(calinski_harabasz_score(X_valid, labels_valid))
    
    return {
        "model_name": model_name,
        "num_clusters": num_clusters,
        "silhouette_score": round(sil, 4),
        "davies_bouldin_index": round(dbi, 4),
        "calinski_harabasz_score": round(chi, 2),
        "valid_evaluation": True
    }


def compare_all_models(
    X: np.ndarray,
    model_results: Dict[str, Dict[str, Any]],
    ground_truth: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Generate a master comparative benchmarking table across all algorithms.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    model_results : dict
        Dictionary of algorithm outputs (K-Means, DBSCAN, Hierarchical, iForest, LOF).
    ground_truth : pd.Series, optional
        Series of ground-truth fraud flags (1 or 0) for academic validation.
        
    Returns:
    --------
    pd.DataFrame
        Comprehensive summary table.
    """
    rows = []
    total_accounts = len(X)
    
    for name, res in model_results.items():
        anomaly_flags = res.get("anomaly_flags")
        if anomaly_flags is None:
            anomaly_flags = res.get("flags")
        if anomaly_flags is None:
            continue
            
        outliers_detected = int(np.sum(anomaly_flags == -1))
        outlier_pct = round((outliers_detected / total_accounts) * 100, 2)
        
        # Clustering metrics if labels exist
        labels = res.get("labels")
        if labels is not None:
            cluster_eval = evaluate_clusters(X, labels, model_name=name)
            sil = cluster_eval.get("silhouette_score")
            dbi = cluster_eval.get("davies_bouldin_index")
            chi = cluster_eval.get("calinski_harabasz_score")
            n_clusters = cluster_eval.get("num_clusters", "-")
        else:
            # Standalone anomaly detectors (Isolation Forest, LOF)
            # Evaluate the binary partition (inliers vs outliers)
            binary_labels = np.array([1 if f == -1 else 0 for f in anomaly_flags])
            if len(set(binary_labels)) > 1:
                sil = round(float(silhouette_score(X, binary_labels, sample_size=min(1500, total_accounts))), 4)
                dbi = round(float(davies_bouldin_score(X, binary_labels)), 4)
                chi = round(float(calinski_harabasz_score(X, binary_labels)), 2)
            else:
                sil, dbi, chi = None, None, None
            n_clusters = 2 # Normal vs Anomaly
            
        row = {
            "Algorithm": name,
            "Type": "Clustering" if "Cluster" in name or name in ["K-Means", "DBSCAN", "Hierarchical"] else "Anomaly Detection",
            "Clusters": n_clusters,
            "Flagged Outliers": outliers_detected,
            "Outlier Rate (%)": outlier_pct,
            "Silhouette Score": sil if sil is not None else "N/A",
            "Davies-Bouldin": dbi if dbi is not None else "N/A",
            "Calinski-Harabasz": chi if chi is not None else "N/A"
        }
        
        # Optional validation against ground truth
        if ground_truth is not None and ground_truth.sum() > 0:
            gt_binary = ground_truth.values
            pred_binary = (anomaly_flags == -1).astype(int)
            
            # Count how many ground truth fraud accounts were successfully flagged
            matched_fraud = int(np.sum((pred_binary == 1) & (gt_binary == 1)))
            recall = round(float(recall_score(gt_binary, pred_binary, zero_division=0) * 100), 2)
            precision = round(float(precision_score(gt_binary, pred_binary, zero_division=0) * 100), 2)
            f1 = round(float(f1_score(gt_binary, pred_binary, zero_division=0) * 100), 2)
            
            row["Fraud Matched"] = f"{matched_fraud}/{int(gt_binary.sum())}"
            row["Recall (%)"] = recall
            row["Precision (%)"] = precision
            row["F1-Score (%)"] = f1
            
        rows.append(row)
        
    df_compare = pd.DataFrame(rows)
    return df_compare
