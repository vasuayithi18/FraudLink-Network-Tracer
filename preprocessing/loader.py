"""
preprocessing/loader.py
-----------------------
Dataset loading, inspection, validation, and cleaning module.
Supports the PaySim financial fraud dataset as well as custom transaction CSVs.
Optimized for large files (e.g., 500MB+ transaction logs) via intelligent sampling.
"""

import io
import os
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np
import pandas as pd

# Expected columns in a standard PaySim dataset
PAYSIM_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud"
]


def load_dataset(
    source: Union[str, io.BytesIO, io.StringIO],
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load a transaction dataset from file path or uploaded buffer.
    
    Optimized for both small CSVs and large files (e.g. 500MB PaySim):
    If the file is on disk and exceeds 20MB, reads an initial candidate chunk
    to avoid exhausting memory, then samples down to the requested size.
    
    Parameters:
    -----------
    source : str or file-like buffer
        Path to CSV file or Streamlit UploadedFile object.
    sample_size : int, optional
        If specified, samples the dataset down to this size for high-performance
        clustering and graph rendering.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        Loaded transaction DataFrame.
    """
    is_path = isinstance(source, str) and os.path.exists(source)
    file_size_mb = (os.path.getsize(source) / (1024 * 1024)) if is_path else 0
    
    # Fast path for large files on disk
    if is_path and file_size_mb > 20 and sample_size:
        # Read a generous candidate pool (e.g. 4x sample_size or at least 40,000 rows)
        candidate_nrows = max(sample_size * 4, 40000)
        df = pd.read_csv(source, nrows=candidate_nrows)
    else:
        df = pd.read_csv(source)
    
    # Clean whitespace in column names
    df.columns = df.columns.str.strip()
    
    if sample_size and len(df) > sample_size:
        # If fraud column is available, ensure fraud cases are proportionally represented
        fraud_col = "isFraud" if "isFraud" in df.columns else None
        if not fraud_col:
            for c in ["fraud", "is_fraud", "class"]:
                if c in df.columns:
                    fraud_col = c
                    break
                    
        if fraud_col and (pd.to_numeric(df[fraud_col], errors="coerce") == 1).sum() > 0:
            is_fraud_mask = pd.to_numeric(df[fraud_col], errors="coerce") == 1
            fraud_df = df[is_fraud_mask]
            non_fraud_df = df[~is_fraud_mask]
            
            # Keep representative fraud cases up to 8% of sample_size
            max_fraud_cases = max(1, int(sample_size * 0.08))
            n_fraud = min(len(fraud_df), max_fraud_cases)
            n_non_fraud = sample_size - n_fraud
            
            sampled_fraud = fraud_df.sample(n=n_fraud, random_state=random_state)
            sampled_non_fraud = non_fraud_df.sample(n=min(n_non_fraud, len(non_fraud_df)), random_state=random_state)
            
            df = pd.concat([sampled_fraud, sampled_non_fraud], ignore_index=True)
            if "step" in df.columns:
                df = df.sort_values(by="step").reset_index(drop=True)
            else:
                df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        else:
            if "step" in df.columns:
                df = df.sample(n=sample_size, random_state=random_state)
                df = df.sort_values(by="step").reset_index(drop=True)
            else:
                df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
            
    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive dataset statistics for exploratory analysis.
    """
    # Sender / Receiver account columns
    sender_col = "nameOrig" if "nameOrig" in df.columns else None
    receiver_col = "nameDest" if "nameDest" in df.columns else None
    
    unique_senders = df[sender_col].nunique() if sender_col else 0
    unique_receivers = df[receiver_col].nunique() if receiver_col else 0
    
    if sender_col and receiver_col:
        all_accounts = set(df[sender_col].dropna().unique()).union(
            set(df[receiver_col].dropna().unique())
        )
        total_unique_accounts = len(all_accounts)
    else:
        total_unique_accounts = 0
        
    missing_series = df.isnull().sum()
    missing_pct = (missing_series / max(len(df), 1)) * 100
    missing_df = pd.DataFrame({
        "Missing Count": missing_series,
        "Missing (%)": missing_pct.round(2)
    })
    
    # Amount stats
    amount_col = "amount" if "amount" in df.columns else None
    if amount_col:
        numeric_amt = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0)
        total_volume = float(numeric_amt.sum())
        avg_amount = float(numeric_amt.mean())
        max_amount = float(numeric_amt.max())
        median_amount = float(numeric_amt.median())
    else:
        total_volume = 0.0
        avg_amount = 0.0
        max_amount = 0.0
        median_amount = 0.0
        
    # Transaction types breakdown
    type_col = "type" if "type" in df.columns else None
    type_counts = df[type_col].value_counts().to_dict() if type_col else {}
    
    # Ground truth (if available) - for evaluation only, NOT for unsupervised training
    has_fraud_label = "isFraud" in df.columns
    if not has_fraud_label:
        for c in ["fraud", "is_fraud"]:
            if c in df.columns:
                has_fraud_label = True
                fraud_series = pd.to_numeric(df[c], errors="coerce") == 1
                fraud_count = int(fraud_series.sum())
                fraud_pct = float(fraud_series.mean() * 100)
                break
        else:
            fraud_count = None
            fraud_pct = None
    else:
        fraud_series = pd.to_numeric(df["isFraud"], errors="coerce") == 1
        fraud_count = int(fraud_series.sum())
        fraud_pct = float(fraud_series.mean() * 100)
    
    return {
        "total_rows": len(df),
        "total_cols": len(df.columns),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
        "missing_summary": missing_df,
        "total_missing_cells": int(missing_series.sum()),
        "duplicate_count": int(df.duplicated().sum()),
        "unique_senders": unique_senders,
        "unique_receivers": unique_receivers,
        "total_unique_accounts": total_unique_accounts,
        "total_volume": total_volume,
        "avg_amount": avg_amount,
        "max_amount": max_amount,
        "median_amount": median_amount,
        "transaction_types": type_counts,
        "has_fraud_label": has_fraud_label,
        "fraud_count": fraud_count,
        "fraud_pct": fraud_pct
    }


def clean_dataset(
    df: pd.DataFrame,
    handle_missing: str = "drop",
    remove_duplicates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Clean dataset by handling missing values and duplicates.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset.
    handle_missing : str
        'drop' - Drop rows with missing values
        'impute' - Impute numeric columns with median, categorical with mode
        'keep' - Do not modify missing values
    remove_duplicates : bool
        Whether to drop exact duplicate rows.
        
    Returns:
    --------
    Tuple[pd.DataFrame, dict]
        (Cleaned DataFrame, Audit report of removed/imputed items)
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)
    duplicates_removed = 0
    missing_handled = 0
    
    if remove_duplicates:
        dups = df_clean.duplicated().sum()
        if dups > 0:
            df_clean = df_clean.drop_duplicates().reset_index(drop=True)
            duplicates_removed = int(dups)
            
    if handle_missing == "drop":
        before = len(df_clean)
        df_clean = df_clean.dropna().reset_index(drop=True)
        missing_handled = before - len(df_clean)
    elif handle_missing == "impute":
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                missing_handled += int(df_clean[col].isnull().sum())
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    median_val = df_clean[col].median()
                    if pd.isna(median_val):
                        median_val = 0.0
                    df_clean[col] = df_clean[col].fillna(median_val)
                else:
                    mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "UNKNOWN"
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    
    audit = {
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "duplicates_removed": duplicates_removed,
        "missing_handled": missing_handled
    }
    
    return df_clean, audit
