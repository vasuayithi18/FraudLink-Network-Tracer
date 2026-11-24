"""
models/anomaly.py
-----------------
Implements dedicated unsupervised anomaly detection algorithms:
1. Isolation Forest (iForest):
   Isolates anomalies by randomly selecting a feature and split value.
   Anomalies (rare, extreme feature values) have noticeably shorter paths in the trees.
2. Local Outlier Factor (LOF):
   Measures the local deviation of density of a given account with respect to its
   k-nearest neighbors. Accounts that have substantially lower density than their
   neighbors are flagged as outliers.

Both models return:
- anomaly_flags: -1 for outlier/anomaly candidate, 1 for inlier/normal
- anomaly_scores: continuous score normalized between 0.0 (safely normal) to 1.0 (most suspicious)
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.03,
    n_estimators: int = 150,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Execute Isolation Forest on standardized account features.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    contamination : float
        Expected proportion of outliers in the data (e.g. 0.01 to 0.05).
    n_estimators : int
        Number of isolation trees in the ensemble.
        
    Returns:
    --------
    dict with:
    - flags: np.ndarray of {-1: anomaly, 1: normal}
    - scores: np.ndarray of normalized anomaly scores in [0, 1] (higher = more anomalous)
    - raw_scores: raw decision function output (lower = more anomalous)
    - outlier_count: total anomalies flagged
    - outlier_pct: percentage of anomalies flagged
    - model: fitted IsolationForest instance
    """
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Flags: -1 for anomaly, 1 for normal
    flags = iso.fit_predict(X)
    
    # Decision function: average anomaly score of base classifiers.
    # The lower the score, the more abnormal. Negative scores represent outliers.
    raw_scores = iso.decision_function(X)
    
    # Normalize score so that 1.0 = most anomalous, 0.0 = completely normal
    min_val, max_val = raw_scores.min(), raw_scores.max()
    if max_val > min_val:
        # Invert so higher score = higher suspicion
        norm_scores = 1.0 - (raw_scores - min_val) / (max_val - min_val)
    else:
        norm_scores = np.zeros_like(raw_scores)
        
    outlier_count = int(np.sum(flags == -1))
    total_samples = len(X)
    
    return {
        "flags": flags,
        "scores": np.round(norm_scores, 4),
        "raw_scores": raw_scores,
        "outlier_count": outlier_count,
        "outlier_pct": round((outlier_count / total_samples) * 100, 2),
        "contamination": contamination,
        "model": iso
    }


def run_lof(
    X: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.03
) -> Dict[str, Any]:
    """
    Execute Local Outlier Factor (LOF) on standardized account features.
    
    Parameters:
    -----------
    X : np.ndarray
        Standardized feature matrix.
    n_neighbors : int
        Number of neighbors to compare local density against.
    contamination : float
        Expected proportion of outliers.
        
    Returns:
    --------
    dict with:
    - flags: np.ndarray of {-1: anomaly, 1: normal}
    - scores: np.ndarray of normalized anomaly scores in [0, 1] (higher = more anomalous)
    - raw_factors: negative outlier factor (close to -1 is normal, large negative is outlier)
    - outlier_count: total anomalies flagged
    - outlier_pct: percentage of anomalies flagged
    - model: fitted LocalOutlierFactor instance
    """
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    
    # Flags: -1 for anomaly, 1 for normal
    flags = lof.fit_predict(X)
    
    # negative_outlier_factor_: The higher, the more normal. Large negative = strong outlier.
    raw_factors = lof.negative_outlier_factor_
    
    # Normalize score so that 1.0 = most anomalous, 0.0 = normal
    min_val, max_val = raw_factors.min(), raw_factors.max()
    if max_val > min_val:
        norm_scores = 1.0 - (raw_factors - min_val) / (max_val - min_val)
    else:
        norm_scores = np.zeros_like(raw_factors)
        
    outlier_count = int(np.sum(flags == -1))
    total_samples = len(X)
    
    return {
        "flags": flags,
        "scores": np.round(norm_scores, 4),
        "raw_factors": raw_factors,
        "outlier_count": outlier_count,
        "outlier_pct": round((outlier_count / total_samples) * 100, 2),
        "n_neighbors": n_neighbors,
        "contamination": contamination,
        "model": lof
    }
