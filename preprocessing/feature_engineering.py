"""
preprocessing/feature_engineering.py
------------------------------------
Extracts account-level behavioral profiles and graph connection metrics from
raw transaction logs.

Financial Crime & AML Context:
In financial forensics, anomalies are best detected by profiling individual
entities (accounts) over time rather than looking at isolated transactions.
This module extracts incoming/outgoing monetary flow, transaction counts,
average amounts, and connection diversity.
"""

from typing import Tuple, List, Optional, Dict
import numpy as np
import pandas as pd


def detect_column_mapping(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """
    Auto-detect common financial transaction column names.
    Supports PaySim natively while accommodating other financial transaction CSVs.
    """
    col_map = {
        "sender": None,
        "receiver": None,
        "amount": None,
        "type": None,
        "step": None,
        "fraud": None
    }
    
    cols_lower = {c.lower(): c for c in df.columns}
    
    # Sender mapping
    for candidate in ["nameorig", "sender", "from_account", "source", "origin", "account_from", "payer"]:
        if candidate in cols_lower:
            col_map["sender"] = cols_lower[candidate]
            break
            
    # Receiver mapping
    for candidate in ["namedest", "receiver", "to_account", "target", "destination", "account_to", "payee"]:
        if candidate in cols_lower:
            col_map["receiver"] = cols_lower[candidate]
            break
            
    # Amount mapping
    for candidate in ["amount", "amt", "value", "transaction_amount", "sum"]:
        if candidate in cols_lower:
            col_map["amount"] = cols_lower[candidate]
            break
            
    # Type mapping
    for candidate in ["type", "txn_type", "transaction_type", "category", "action"]:
        if candidate in cols_lower:
            col_map["type"] = cols_lower[candidate]
            break
            
    # Step/Time mapping
    for candidate in ["step", "timestamp", "date", "time", "datetime", "hour"]:
        if candidate in cols_lower:
            col_map["step"] = cols_lower[candidate]
            break
            
    # Ground truth fraud flag mapping (for evaluation only, NEVER for training)
    for candidate in ["isfraud", "fraud", "is_fraud", "class", "label", "is_anomaly"]:
        if candidate in cols_lower:
            col_map["fraud"] = cols_lower[candidate]
            break
            
    return col_map


def extract_account_features(
    df: pd.DataFrame,
    sender_col: str = "nameOrig",
    receiver_col: str = "nameDest",
    amount_col: str = "amount",
    type_col: str = "type",
    oldbal_orig_col: str = "oldbalanceOrg",
    newbal_orig_col: str = "newbalanceOrig"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transform raw transaction logs into an Account Profile feature matrix.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction logs.
    sender_col : str
        Column for transaction originator.
    receiver_col : str
        Column for transaction destination.
    amount_col : str
        Column for transaction amount.
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        - features_df: DataFrame indexed by account_id containing unsupervised features.
        - ground_truth: Series of ground-truth fraud flags (if available), NEVER to be
          included in unsupervised model training.
    """
    df_tx = df.copy()
    
    # Validate critical columns
    if sender_col not in df_tx.columns or receiver_col not in df_tx.columns or amount_col not in df_tx.columns:
        auto_map = detect_column_mapping(df_tx)
        sender_col = auto_map["sender"] or sender_col
        receiver_col = auto_map["receiver"] or receiver_col
        amount_col = auto_map["amount"] or amount_col
        
    # Ensure amount is numeric
    df_tx[amount_col] = pd.to_numeric(df_tx[amount_col], errors="coerce").fillna(0.0)
    
    # 1. Outgoing Metrics (when account is sender)
    # Total money sent, outgoing count, avg sent, max sent, unique destinations
    outgoing = df_tx.groupby(sender_col).agg(
        total_sent=(amount_col, "sum"),
        txn_sent_count=(amount_col, "count"),
        avg_sent_amount=(amount_col, "mean"),
        max_sent_amount=(amount_col, "max"),
        unique_dest_count=(receiver_col, "nunique")
    )
    
    # Check for balance draining (balance reaches 0 after transaction)
    if oldbal_orig_col in df_tx.columns and newbal_orig_col in df_tx.columns:
        df_tx[oldbal_orig_col] = pd.to_numeric(df_tx[oldbal_orig_col], errors="coerce").fillna(0.0)
        df_tx[newbal_orig_col] = pd.to_numeric(df_tx[newbal_orig_col], errors="coerce").fillna(0.0)
        df_tx["is_drained"] = (
            (df_tx[newbal_orig_col] == 0) & (df_tx[oldbal_orig_col] > 0)
        ).astype(int)
        drained_counts = df_tx.groupby(sender_col)["is_drained"].sum().rename("balance_drain_count")
        outgoing = outgoing.join(drained_counts, how="left")
    else:
        outgoing["balance_drain_count"] = 0
        
    # High risk transaction count (TRANSFER and CASH_OUT are the primary vehicles for fraud in PaySim)
    if type_col in df_tx.columns:
        df_tx["is_high_risk_type"] = df_tx[type_col].astype(str).str.upper().isin(["TRANSFER", "CASH_OUT"]).astype(int)
        high_risk_counts = df_tx.groupby(sender_col)["is_high_risk_type"].sum().rename("high_risk_txn_count")
        outgoing = outgoing.join(high_risk_counts, how="left")
    else:
        outgoing["high_risk_txn_count"] = 0

    # 2. Incoming Metrics (when account is receiver)
    # Total money received, incoming count, avg received, max received, unique senders
    incoming = df_tx.groupby(receiver_col).agg(
        total_received=(amount_col, "sum"),
        txn_received_count=(amount_col, "count"),
        avg_received_amount=(amount_col, "mean"),
        max_received_amount=(amount_col, "max"),
        unique_orig_count=(sender_col, "nunique")
    )
    
    # 3. Combine Outgoing and Incoming into Unified Account Registry
    accounts_index = sorted(list(set(outgoing.index).union(set(incoming.index))))
    accounts_df = pd.DataFrame(index=accounts_index)
    accounts_df.index.name = "account_id"
    
    # Merge both perspectives
    accounts_df = accounts_df.join(outgoing, how="left").fillna(0.0)
    accounts_df = accounts_df.join(incoming, how="left").fillna(0.0)
    
    # 4. Synthesize Flow & Velocity Behavioral Features
    # Total transaction count
    accounts_df["total_txn_count"] = accounts_df["txn_sent_count"] + accounts_df["txn_received_count"]
    
    # Total money throughput
    accounts_df["total_turnover"] = accounts_df["total_sent"] + accounts_df["total_received"]
    
    # Net flow (Received - Sent): Mule accounts often have net flow close to zero (inflow ~= outflow)
    accounts_df["net_flow"] = accounts_df["total_received"] - accounts_df["total_sent"]
    
    # Total unique connected entities (Degree in the transaction graph)
    accounts_df["total_connected_accounts"] = accounts_df["unique_dest_count"] + accounts_df["unique_orig_count"]
    
    # Ratio of outflow to total flow: 1.0 means pure drainer/sender, 0.0 means pure aggregator/sink
    accounts_df["outflow_ratio"] = accounts_df["total_sent"] / (accounts_df["total_turnover"] + 1e-5)
    
    # Average transaction size overall
    accounts_df["overall_avg_amount"] = accounts_df["total_turnover"] / (accounts_df["total_txn_count"] + 1e-5)
    
    # 5. Extract Ground-Truth Fraud Labels (ONLY for evaluation/reference, NOT features)
    ground_truth = pd.Series(0, index=accounts_df.index, name="is_ground_truth_fraud")
    fraud_col = "isFraud" if "isFraud" in df_tx.columns else None
    if not fraud_col:
        auto_map = detect_column_mapping(df_tx)
        fraud_col = auto_map.get("fraud")
        
    if fraud_col and fraud_col in df_tx.columns:
        fraud_mask = pd.to_numeric(df_tx[fraud_col], errors="coerce") == 1
        fraud_senders = df_tx[fraud_mask][sender_col].dropna().unique()
        fraud_receivers = df_tx[fraud_mask][receiver_col].dropna().unique()
        fraud_all = set(fraud_senders).union(set(fraud_receivers))
        ground_truth.loc[ground_truth.index.isin(fraud_all)] = 1
        
    return accounts_df, ground_truth


def get_feature_names() -> List[str]:
    """Return the complete ordered list of all 18 numeric behavioral feature columns."""
    return [
        "total_sent",
        "total_received",
        "txn_sent_count",
        "txn_received_count",
        "avg_sent_amount",
        "avg_received_amount",
        "max_sent_amount",
        "max_received_amount",
        "unique_dest_count",
        "unique_orig_count",
        "total_txn_count",
        "total_turnover",
        "net_flow",
        "total_connected_accounts",
        "outflow_ratio",
        "overall_avg_amount",
        "balance_drain_count",
        "high_risk_txn_count"
    ]
