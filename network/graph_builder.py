"""
network/graph_builder.py
------------------------
Constructs directed transaction graphs (NetworkX DiGraph) from financial logs.
Integrates account behavioral attributes and unsupervised anomaly scores from Phase 3.

Key Graph Concepts (Viva Prep):
- Nodes: Accounts (customers 'C...', merchants 'M...').
- Directed Edges: Financial transactions from sender (nameOrig) -> receiver (nameDest).
- Edge Attributes: Transaction amount ($), type (TRANSFER, CASH_OUT, etc.), time step.
- Node Attributes: In-degree (incoming txns), out-degree (outgoing txns),
  total sent, total received, net flow, anomaly candidate status, PageRank centrality.
"""

from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
import networkx as nx


def build_transaction_graph(
    df: pd.DataFrame,
    anomaly_candidates: Optional[Set[str]] = None,
    anomaly_scores: Optional[Dict[str, float]] = None,
    min_amount: float = 0.0,
    txn_types: Optional[List[str]] = None,
    time_range: Optional[tuple] = None,
    max_edges: int = 3000
) -> nx.DiGraph:
    """
    Construct a directed transaction graph from a DataFrame of transactions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Transaction records (containing nameOrig, nameDest, amount, type, step).
    anomaly_candidates : set of str, optional
        Account IDs flagged as anomaly candidates by Phase 3 ML models.
    anomaly_scores : dict of str -> float, optional
        Mapping from account_id to continuous anomaly score in [0.0, 1.0].
    min_amount : float
        Filter: only include transactions with amount >= min_amount.
    txn_types : list of str, optional
        Filter: only include specific transaction types (e.g. ['TRANSFER', 'CASH_OUT']).
    time_range : tuple of (int, int), optional
        Filter: only include steps within [min_step, max_step].
    max_edges : int
        Maximum number of edges to construct for high-performance rendering.
        
    Returns:
    --------
    nx.DiGraph
        Enriched directed network graph.
    """
    df_filtered = df.copy()
    
    # 1. Apply Filtering for Performance & Focused Analysis
    sender_col = "nameOrig" if "nameOrig" in df_filtered.columns else "sender"
    receiver_col = "nameDest" if "nameDest" in df_filtered.columns else "receiver"
    amount_col = "amount" if "amount" in df_filtered.columns else "value"
    type_col = "type" if "type" in df_filtered.columns else None
    step_col = "step" if "step" in df_filtered.columns else None
    
    # Amount filter
    if min_amount > 0 and amount_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[amount_col] >= min_amount]
        
    # Transaction type filter
    if txn_types and type_col and type_col in df_filtered.columns:
        df_filtered = df_filtered[df_filtered[type_col].isin(txn_types)]
        
    # Time window filter
    if time_range and step_col and step_col in df_filtered.columns:
        min_s, max_s = time_range
        df_filtered = df_filtered[(df_filtered[step_col] >= min_s) & (df_filtered[step_col] <= max_s)]
        
    # Cap total edges for performance (prioritize higher value or recent transactions)
    if len(df_filtered) > max_edges:
        df_filtered = df_filtered.sort_values(by=amount_col, ascending=False).head(max_edges)
        
    G = nx.DiGraph()
    
    # 2. Add Directed Edges with Attributes
    for _, row in df_filtered.iterrows():
        src = str(row[sender_col])
        dst = str(row[receiver_col])
        amt = float(row[amount_col])
        ttype = str(row[type_col]) if type_col and type_col in row else "TRANSACTION"
        step = int(row[step_col]) if step_col and step_col in row else 1
        
        # If an edge already exists, aggregate amounts or store multi-edge info
        if G.has_edge(src, dst):
            G[src][dst]["amount"] += amt
            G[src][dst]["count"] += 1
            G[src][dst]["types"].add(ttype)
        else:
            G.add_edge(
                src,
                dst,
                amount=amt,
                count=1,
                types={ttype},
                type=ttype,
                step=step
            )
            
    # Convert sets to comma-separated strings for serialization
    for u, v, d in G.edges(data=True):
        if "types" in d and isinstance(d["types"], set):
            d["type_str"] = ", ".join(sorted(list(d["types"])))
            
    # 3. Compute and Attach Node Attributes
    anomaly_set = set(anomaly_candidates) if anomaly_candidates else set()
    scores_dict = anomaly_scores or {}
    
    # Compute incoming and outgoing volumes per node
    node_sent = {}
    node_received = {}
    for u, v, d in G.edges(data=True):
        node_sent[u] = node_sent.get(u, 0.0) + d["amount"]
        node_received[v] = node_received.get(v, 0.0) + d["amount"]
        
    # Compute PageRank centrality (indicates structural importance in money flow)
    try:
        pagerank = nx.pagerank(G, weight="amount", max_iter=100)
    except Exception:
        pagerank = {n: 0.0 for n in G.nodes()}
        
    for node in G.nodes():
        sent = node_sent.get(node, 0.0)
        recv = node_received.get(node, 0.0)
        is_anom = node in anomaly_set
        score = scores_dict.get(node, 1.0 if is_anom else 0.0)
        
        G.nodes[node]["total_sent"] = round(sent, 2)
        G.nodes[node]["total_received"] = round(recv, 2)
        G.nodes[node]["net_flow"] = round(recv - sent, 2)
        G.nodes[node]["in_degree"] = G.in_degree(node)
        G.nodes[node]["out_degree"] = G.out_degree(node)
        G.nodes[node]["total_degree"] = G.degree(node)
        G.nodes[node]["is_anomaly"] = is_anom
        G.nodes[node]["anomaly_score"] = round(score, 4)
        G.nodes[node]["pagerank"] = round(pagerank.get(node, 0.0), 6)
        G.nodes[node]["is_merchant"] = node.startswith("M")
        
    return G


def get_graph_summary(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Extract comprehensive topological summary statistics from the graph.
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "total_volume": 0.0,
            "density": 0.0, "weakly_connected_components": 0,
            "strongly_connected_components": 0, "anomaly_nodes_count": 0
        }
        
    total_volume = sum(d.get("amount", 0.0) for _, _, d in G.edges(data=True))
    density = nx.density(G)
    wcc = nx.number_weakly_connected_components(G)
    scc = nx.number_strongly_connected_components(G)
    
    anomaly_count = sum(1 for _, d in G.nodes(data=True) if d.get("is_anomaly", False))
    
    # Find top 5 hub accounts by degree
    sorted_by_degree = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:5]
    top_hubs = [{"account": n, "connections": deg} for n, deg in sorted_by_degree]
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "total_volume": round(total_volume, 2),
        "density": round(density, 6),
        "weakly_connected_components": wcc,
        "strongly_connected_components": scc,
        "anomaly_nodes_count": anomaly_count,
        "anomaly_pct": round((anomaly_count / max(num_nodes, 1)) * 100, 2),
        "top_hubs": top_hubs
    }
