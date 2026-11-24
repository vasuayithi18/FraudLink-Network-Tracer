"""
reports/reporter.py
-------------------
Forensic Investigation Reporting and Explainability Engine:
- Generates structured suspect account summaries.
- Produces plain-language explanations for why an account was flagged (Rule + ML Consensus).
- Computes Risk Tiers (CRITICAL, HIGH, MEDIUM).
- Exports investigation audit reports to CSV.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def generate_forensic_investigation_report(
    features_df: pd.DataFrame,
    model_results: Dict[str, Dict[str, Any]],
    top_n: int = 100
) -> pd.DataFrame:
    """
    Synthesize ML model flags and behavioral features into an explainable
    forensic investigation report.
    
    Parameters:
    -----------
    features_df : pd.DataFrame
        Unscaled raw behavioral features indexed by account_id.
    model_results : dict
        Outputs from clustering and anomaly models.
        
    Returns:
    --------
    pd.DataFrame
        Investigator-ready audit table.
    """
    account_ids = list(features_df.index)
    N = len(account_ids)
    
    # Collect votes from each algorithm
    model_votes = {acc: [] for acc in account_ids}
    score_map = {acc: 0.0 for acc in account_ids}
    
    for name, res in model_results.items():
        flags = res.get("anomaly_flags")
        if flags is None:
            flags = res.get("flags")
        if flags is None:
            continue
            
        scores = res.get("scores")
        for i, acc in enumerate(account_ids):
            if i < len(flags) and flags[i] == -1:
                model_votes[acc].append(name)
            if scores is not None and i < len(scores):
                # Keep the highest anomaly score across models
                score_map[acc] = max(score_map[acc], float(scores[i]))
                
    # Filter to accounts flagged by at least one model
    flagged_accounts = [acc for acc in account_ids if len(model_votes[acc]) > 0]
    
    report_rows = []
    
    for acc in flagged_accounts:
        row = features_df.loc[acc]
        votes = model_votes[acc]
        vote_count = len(votes)
        max_score = score_map.get(acc, 0.0)
        
        total_sent = float(row.get("total_sent", 0.0))
        total_recv = float(row.get("total_received", 0.0))
        total_turnover = float(row.get("total_turnover", 0.0))
        net_flow = float(row.get("net_flow", 0.0))
        drain_count = int(row.get("balance_drain_count", 0))
        high_risk_txns = int(row.get("high_risk_txn_count", 0))
        connected_count = int(row.get("total_connected_accounts", 0))
        outflow_ratio = float(row.get("outflow_ratio", 0.0))
        
        # Rule-based forensic explanation generator
        reasons = []
        if drain_count > 0:
            reasons.append(f"Account balance emptied to $0 after outflow ({drain_count} time(s))")
        if total_sent > 200000:
            reasons.append(f"Large capital outflow (${total_sent:,.2f})")
        if total_recv > 200000 and total_sent > 150000 and abs(net_flow) < 50000:
            reasons.append("Pass-through money mule behavior (funds rapidly aggregated and forwarded)")
        if connected_count >= 6:
            reasons.append(f"Abnormally high counterparty connectivity ({connected_count} unique partners)")
        if high_risk_txns >= 2:
            reasons.append(f"Heavy utilization of high-risk vehicles ({high_risk_txns} TRANSFER/CASH_OUT transactions)")
        if outflow_ratio > 0.95 and total_sent > 50000:
            reasons.append("Extreme unidirectional fund drain (outflow > 95% of throughput)")
        if vote_count >= 2:
            reasons.append(f"Multi-model ML consensus ({vote_count} algorithms in agreement)")
            
        if not reasons:
            reasons.append("Statistical feature outlier in multidimensional behavioral space")
            
        primary_reason = " | ".join(reasons)
        
        # Risk Tier Classification
        if vote_count >= 2 or max_score >= 0.85 or drain_count > 0:
            risk_tier = "CRITICAL"
        elif vote_count == 1 and (max_score >= 0.65 or total_turnover > 100000):
            risk_tier = "HIGH"
        else:
            risk_tier = "MEDIUM"
            
        report_rows.append({
            "Account ID": acc,
            "Risk Tier": risk_tier,
            "Max Anomaly Score": round(max_score, 4),
            "Algorithms Flagging": ", ".join(votes),
            "Total Sent ($)": round(total_sent, 2),
            "Total Received ($)": round(total_recv, 2),
            "Net Flow ($)": round(net_flow, 2),
            "Connected Accounts": connected_count,
            "Balance Drains": drain_count,
            "Primary Forensic Explanations": primary_reason
        })
        
    df_report = pd.DataFrame(report_rows)
    
    if not df_report.empty:
        # Sort by Risk Tier (CRITICAL -> HIGH -> MEDIUM) then Anomaly Score descending
        tier_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        df_report["tier_rank"] = df_report["Risk Tier"].map(tier_order)
        df_report = df_report.sort_values(by=["tier_rank", "Max Anomaly Score"], ascending=[True, False]).drop(columns=["tier_rank"])
        df_report = df_report.head(top_n).reset_index(drop=True)
        
    return df_report


def export_report_to_csv(df_report: pd.DataFrame) -> str:
    """Export the report DataFrame to a CSV string."""
    return df_report.to_csv(index=False)
