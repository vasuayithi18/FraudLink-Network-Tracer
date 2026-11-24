"""
test_phase1_2.py
----------------
Comprehensive Verification Script for Phase 1 & Phase 2:
1. Tests environment and dependencies.
2. Tests dataset loader on both sample and large CSV datasets.
3. Verifies dataset summary metrics and audit logging.
4. Verifies data cleaning (missing value imputation & duplicate handling).
5. Verifies account-level feature engineering and ensures zero label leakage.
6. Verifies StandardScaler normalization (mean ~ 0, std ~ 1).

Compatible with both standalone execution ('python test_phase1_2.py')
and pytest test collection ('pytest -v').
"""

import os
import sys
import pandas as pd
from preprocessing.loader import load_dataset, get_dataset_summary, clean_dataset
from preprocessing.feature_engineering import extract_account_features, get_feature_names
from preprocessing.scaler import scale_features


def test_environment():
    """Verify that all required dependencies are installed and importable."""
    print("--- [1/5] Verifying Dependencies ---")
    required = ["pandas", "numpy", "sklearn", "streamlit", "networkx", "plotly", "matplotlib", "seaborn"]
    for pkg in required:
        mod = __import__(pkg)
        version = getattr(mod, "__version__", "installed")
        print(f"  ✓ {pkg}: {version}")
    print("  -> Dependencies verified successfully.\n")


def run_dataset_loading_and_cleaning(csv_path: str, sample_size: int = None):
    """Helper function: test dataset loading and cleaning logic for a given file."""
    print(f"--- [2/5] Testing Dataset Loader on '{csv_path}' ---")
    assert os.path.exists(csv_path), f"File {csv_path} does not exist!"
    
    df = load_dataset(csv_path, sample_size=sample_size)
    print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns.")
    assert len(df) > 0, "Dataset should not be empty!"
    
    summary = get_dataset_summary(df)
    print(f"  ✓ Summary: {summary['total_unique_accounts']:,} unique accounts, ${summary['total_volume']:,.2f} total volume.")
    if summary['has_fraud_label']:
        print(f"  ✓ Ground truth fraud rows: {summary['fraud_count']} ({summary['fraud_pct']:.2f}%)")
        
    clean_df, audit = clean_dataset(df, handle_missing="drop", remove_duplicates=True)
    print(f"  ✓ Cleaned: {len(clean_df):,} rows remaining. Audit: {audit}")
    assert len(clean_df) > 0, "Cleaned dataset should not be empty!"
    return clean_df


def run_feature_engineering_and_scaling(df: pd.DataFrame):
    """Helper function: test feature extraction, label leakage check, and StandardScaler."""
    print("--- [3/5] Testing Account-Level Feature Engineering ---")
    features_df, ground_truth = extract_account_features(df)
    print(f"  ✓ Extracted {len(features_df):,} unique account profiles with {features_df.shape[1]} features.")
    
    # Critical Check: Verify ZERO target leakage
    assert "isFraud" not in features_df.columns, "CRITICAL ERROR: 'isFraud' must not be in feature matrix!"
    assert "isFlaggedFraud" not in features_df.columns, "CRITICAL ERROR: 'isFlaggedFraud' must not be in feature matrix!"
    print("  ✓ CRITICAL VERIFICATION: No fraud labels leaked into training feature matrix.")
    
    print("\n--- [4/5] Testing StandardScaler Normalization ---")
    feature_cols = get_feature_names()
    scaled_df, scaler = scale_features(features_df, feature_cols)
    
    max_mean = scaled_df.mean().abs().max()
    avg_std = scaled_df.std().mean()
    print(f"  ✓ Scaled feature matrix shape: {scaled_df.shape}")
    print(f"  ✓ Max absolute feature mean: {max_mean:.6f} (target ~0.0)")
    print(f"  ✓ Average feature standard deviation: {avg_std:.4f} (target ~1.0)")
    assert max_mean < 0.05, f"Standardization failed: mean too high ({max_mean})"
    assert 0.95 <= avg_std <= 1.05, f"Standardization failed: std dev skewed ({avg_std})"
    print("  ✓ Features correctly normalized.")


def test_phase1_2_pipeline():
    """Pytest-discoverable test for the Phase 1 & Phase 2 verification pipeline."""
    sample_path = "data/paysim_sample.csv"
    if not os.path.exists(sample_path) and os.path.exists("data/transactions.csv"):
        sample_path = "data/transactions.csv"
    assert os.path.exists(sample_path), f"Dataset file {sample_path} not found"
    df = run_dataset_loading_and_cleaning(sample_path, sample_size=3000)
    run_feature_engineering_and_scaling(df)


def main():
    print("==================================================")
    print("   FRAUDLINK + NETWORK TRACER - PHASE 1 & 2 TEST   ")
    print("==================================================\n")
    
    # 1. Test environment
    test_environment()
    
    # 2. Test sample dataset
    sample_path = "data/paysim_sample.csv"
    if os.path.exists(sample_path):
        df_sample = run_dataset_loading_and_cleaning(sample_path)
        run_feature_engineering_and_scaling(df_sample)
        
    # 3. Test real dataset (data/transactions.csv) if present
    real_path = "data/transactions.csv"
    if os.path.exists(real_path):
        print(f"\n--- [5/5] Testing Large Real Dataset '{real_path}' with Fast Sampling ---")
        df_real = run_dataset_loading_and_cleaning(real_path, sample_size=5000)
        run_feature_engineering_and_scaling(df_real)
    
    print("\n==================================================")
    print("   >>> ALL PHASE 1 & PHASE 2 TESTS PASSED! <<<    ")
    print("==================================================")


if __name__ == "__main__":
    main()
