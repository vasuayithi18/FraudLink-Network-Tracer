"""
models/clustering.py
-------------------
Implements unsupervised clustering algorithms for financial transaction accounts:
1. K-Means Clustering (with Elbow method and Silhouette evaluation)
2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
3. Hierarchical / Agglomerative Clustering (with linkage matrix for dendrograms)

All algorithms operate on standardized account-level behavioral features.
Fraud labels (isFraud, isFlaggedFraud) are strictly excluded from all training.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage


def compute_elbow_curve(
    X: np.ndarray,
    k_range: range = range(2, 10),
    random_state: int = 42
) -> Dict[str, List[float]]:
    """
    Compute Inertia (Within-Cluster Sum of Squares) and Silhouette Scores
    across a range of K values to find the optimal number of clusters.
    
    Viva Concept:
    - Inertia measures how tightly packed the clusters are (lower is better,
      but monotonically decreases as k increases).
    - The 'Elbow' is the point of diminishing returns where inertia drop slows.
    - Silhouette score ranges from -1 to +1 (higher is better).
    """
    inertias = []
    silhouettes = []
    k_values = list(k_range)
    
    # Subsample if X is large for fast interactive response in Streamlit
    if len(X) > 2500:
        np.random.seed(random_state)
        idx = np.random.choice(len(X), size=2500, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X
        
    for k in k_values:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = km.fit_predict(X_eval)
        inertias.append(float(km.inertia_))
        
        # Silhouette score requires at least 2 distinct clusters
        if len(set(labels)) > 1:
            sil = float(silhouette_score(X_eval, labels, sample_size=min(1000, len(X_eval))))
            silhouettes.append(sil)
        else:
            silhouettes.append(0.0)
            
    return {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes
    }


def run_kmeans(
    X: np.ndarray,
    n_clusters: int = 4,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute K-Means Clustering on normalized account features.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Number of clusters (centroids) to fit.
        
    Returns:
    --------
    dict with keys:
    - labels: Cluster assignments (0 to n_clusters-1)
    - centroids: Coordinates of cluster centers
    - inertia: Total within-cluster sum of squared distances
    - model: Fitted KMeans instance
    """
    km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, random_state=random_state)
    labels = km.fit_predict(X)
    
    # Identify minority clusters as potential anomaly candidates
    cluster_counts = pd.Series(labels).value_counts().to_dict()
    total_samples = len(X)
    
    # Flag accounts belonging to clusters comprising < 3% of the population
    minority_clusters = [c for c, count in cluster_counts.items() if (count / total_samples) < 0.03]
    anomaly_flags = np.array([-1 if lbl in minority_clusters else 1 for lbl in labels])
    
    return {
        "labels": labels,
        "centroids": km.cluster_centers_,
        "inertia": float(km.inertia_),
        "cluster_counts": cluster_counts,
        "minority_clusters": minority_clusters,
        "anomaly_flags": anomaly_flags,
        "model": km
    }


def run_dbscan(
    X: np.ndarray,
    eps: float = 1.2,
    min_samples: int = 5
) -> Dict[str, Any]:
    """
    Execute DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Viva Concept:
    DBSCAN discovers clusters of arbitrary shape and automatically identifies
    anomalies as 'noise points' (label = -1) without requiring a pre-specified K.
    - Core points: Have at least min_samples within distance eps.
    - Border points: Within eps of a core point but have fewer than min_samples.
    - Noise points: Neither core nor border. These represent isolated, anomalous accounts!
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    eps : float
        Maximum distance between two samples for one to be considered in neighborhood.
    min_samples : int
        Number of samples in neighborhood for a point to be considered core.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X)
    
    # Identify core, border, and noise samples
    core_sample_indices = set(db.core_sample_indices_)
    total_samples = len(X)
    
    point_types = []
    for i in range(total_samples):
        if i in core_sample_indices:
            point_types.append("Core")
        elif labels[i] == -1:
            point_types.append("Noise (Anomaly)")
        else:
            point_types.append("Border")
            
    point_types = np.array(point_types)
    noise_count = int(np.sum(labels == -1))
    unique_clusters = set(labels) - {-1}
    num_clusters = len(unique_clusters)
    
    # Binary anomaly flags: -1 for noise (outlier), 1 for clustered (normal)
    anomaly_flags = np.array([-1 if lbl == -1 else 1 for lbl in labels])
    
    return {
        "labels": labels,
        "num_clusters": num_clusters,
        "noise_count": noise_count,
        "noise_pct": round((noise_count / total_samples) * 100, 2),
        "point_types": point_types,
        "anomaly_flags": anomaly_flags,
        "eps": eps,
        "min_samples": min_samples,
        "model": db
    }


def run_hierarchical(
    X: np.ndarray,
    n_clusters: int = 4,
    linkage_method: str = "ward",
    sample_limit: int = 2500,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute Agglomerative Hierarchical Clustering.
    
    Viva Concept:
    Bottom-up clustering where each observation starts in its own cluster and
    pairs of clusters are merged as one moves up the hierarchy.
    - Ward linkage minimizes total within-cluster variance (best for spherical clusters).
    - Complete linkage uses maximum distance between clusters.
    - Average linkage uses average distance between clusters.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Number of clusters to extract.
    linkage_method : str
        'ward', 'complete', 'average', or 'single'.
    sample_limit : int
        Agglomerative clustering is O(N^2) memory. If dataset exceeds sample_limit,
        we sample for safety and compute clusters.
    """
    N = len(X)
    if N > sample_limit:
        np.random.seed(random_state)
        sub_indices = np.random.choice(N, size=sample_limit, replace=False)
        X_fit = X[sub_indices]
        sampled = True
    else:
        sub_indices = np.arange(N)
        X_fit = X
        sampled = False
        
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    sub_labels = model.fit_predict(X_fit)
    
    # Map back to full dataset if sampled
    if sampled:
        full_labels = np.full(N, -1, dtype=int)
        full_labels[sub_indices] = sub_labels
    else:
        full_labels = sub_labels
        
    cluster_counts = pd.Series(sub_labels).value_counts().to_dict()
    # Minority clusters flagged as anomaly candidates
    minority_clusters = [c for c, count in cluster_counts.items() if (count / len(X_fit)) < 0.03]
    anomaly_flags = np.array([-1 if lbl in minority_clusters else 1 for lbl in full_labels])
    
    return {
        "labels": full_labels,
        "sub_labels": sub_labels,
        "sub_indices": sub_indices,
        "num_clusters": n_clusters,
        "linkage": linkage_method,
        "cluster_counts": cluster_counts,
        "minority_clusters": minority_clusters,
        "anomaly_flags": anomaly_flags,
        "sampled": sampled,
        "model": model
    }


def compute_hierarchical_linkage(
    X: np.ndarray,
    method: str = "ward",
    sample_size: int = 150,
    random_state: int = 42
) -> np.ndarray:
    """
    Compute linkage matrix for dendrogram visualization on a representative sample.
    (Plotting all 3,000+ leaves in a dendrogram is unreadable; 100-200 leaves is optimal).
    """
    if len(X) > sample_size:
        np.random.seed(random_state)
        sample_idx = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        
    return linkage(X_sample, method=method)
