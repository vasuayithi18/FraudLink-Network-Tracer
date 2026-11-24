"""
preprocessing/scaler.py
-----------------------
Standardizes extracted features using scikit-learn's StandardScaler.

Why Standard Scaling is Essential in Financial Anomaly Detection:
Monetary values ($10 to $1,000,000) operate on orders of magnitude higher scales
than transaction counts (1 to 20) or connection counts (1 to 10). Without
standardization, distance-based algorithms (K-Means, DBSCAN, Hierarchical, LOF)
would be entirely dominated by the monetary sum and ignore transaction velocity
and network topology.
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features(
    features_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features to zero mean and unit variance.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        DataFrame containing raw engineered features.
    feature_cols : list of str, optional
        Subset of columns to scale. If None, all numeric columns are scaled.
        
    Returns:
    --------
    Tuple[pd.DataFrame, StandardScaler]
        - scaled_df: DataFrame with standardized numeric columns, preserving index.
        - scaler: Fitted StandardScaler instance.
    """
    df_to_scale = features_df.copy()
    
    if feature_cols is None:
        feature_cols = [c for c in df_to_scale.columns if pd.api.types.is_numeric_dtype(df_to_scale[c])]
        
    # Replace any infinite values with NaN then fill with median
    clean_matrix = df_to_scale[feature_cols].replace([np.inf, -np.inf], np.nan)
    clean_matrix = clean_matrix.fillna(clean_matrix.median())
    
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(clean_matrix)
    
    scaled_df = pd.DataFrame(
        scaled_values,
        index=features_df.index,
        columns=feature_cols
    )
    
    return scaled_df, scaler
