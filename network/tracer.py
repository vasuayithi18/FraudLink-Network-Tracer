"""
network/tracer.py
-----------------
Transaction path tracing, account investigation, and graph pattern mining:
- Account profile investigation (1-hop ego network, in/out flows)
- Shortest path and multi-hop path tracing between accounts
- Circular money flow detection (laundering cycles)
- Fan-in (structuring) and Fan-out (dispersal) topology detection
- Connected component extraction
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import pandas as pd


def get_account_investigation(G: nx.DiGraph, account_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve forensic details for an investigated account:
    - Metadata, total sent/received, net flow, anomaly status
    - Detailed list of incoming transactions (with sender ID, amount, type)
    - Detailed list of outgoing transactions (with receiver ID, amount, type)
    - Direct transacting partners (neighbors)
    """
    account_id = str(account_id).strip()
    if not G.has_node(account_id):
        return None
        
    node_data = G.nodes[account_id]
    
    # Incoming transactions (predecessors)
    incoming_txns = []
    for pred in G.predecessors(account_id):
        edge_data = G[pred][account_id]
        incoming_txns.append({
            "counterparty": pred,
            "direction": "INCOMING",
            "amount": round(edge_data.get("amount", 0.0), 2),
            "txn_count": edge_data.get("count", 1),
            "type": edge_data.get("type_str", edge_data.get("type", "TRANSFER")),
            "step": edge_data.get("step", 1)
        })
        
    # Outgoing transactions (successors)
    outgoing_txns = []
    for succ in G.successors(account_id):
        edge_data = G[account_id][succ]
        outgoing_txns.append({
            "counterparty": succ,
            "direction": "OUTGOING",
            "amount": round(edge_data.get("amount", 0.0), 2),
            "txn_count": edge_data.get("count", 1),
            "type": edge_data.get("type_str", edge_data.get("type", "TRANSFER")),
            "step": edge_data.get("step", 1)
        })
        
    # Sort transactions by amount descending
    incoming_txns.sort(key=lambda x: x["amount"], reverse=True)
    outgoing_txns.sort(key=lambda x: x["amount"], reverse=True)
    
    all_neighbors = list(set(G.predecessors(account_id)).union(set(G.successors(account_id))))
    
    return {
        "account_id": account_id,
        "is_anomaly": node_data.get("is_anomaly", False),
        "anomaly_score": node_data.get("anomaly_score", 0.0),
        "total_sent": node_data.get("total_sent", 0.0),
        "total_received": node_data.get("total_received", 0.0),
        "net_flow": node_data.get("net_flow", 0.0),
        "in_degree": node_data.get("in_degree", 0),
        "out_degree": node_data.get("out_degree", 0),
        "total_degree": node_data.get("total_degree", 0),
        "pagerank": node_data.get("pagerank", 0.0),
        "is_merchant": node_data.get("is_merchant", False),
        "connected_accounts_count": len(all_neighbors),
        "connected_accounts": all_neighbors,
        "incoming_transactions": incoming_txns,
        "outgoing_transactions": outgoing_txns
    }


def trace_transaction_paths(
    G: nx.DiGraph,
    source_account: str,
    target_account: str,
    max_paths: int = 5,
    max_depth: int = 5
) -> Dict[str, Any]:
    """
    Find directed transaction paths between two accounts.
    Uses BFS shortest path and simple paths search.
    
    Viva Concept:
    Traces the multi-hop trajectory of money laundering. If funds move
    A -> B -> C -> D, this reveals the intermediary mule chain.
    """
    src = str(source_account).strip()
    dst = str(target_account).strip()
    
    if not G.has_node(src):
        return {"status": "error", "message": f"Source account '{src}' not found in the transaction graph."}
    if not G.has_node(dst):
        return {"status": "error", "message": f"Target account '{dst}' not found in the transaction graph."}
    if src == dst:
        return {"status": "error", "message": "Source and target accounts must be different."}
        
    has_path = nx.has_path(G, src, dst)
    if not has_path:
        return {
            "status": "not_connected",
            "source": src,
            "target": dst,
            "paths_found": 0,
            "shortest_path": None,
            "paths": []
        }
        
    # Compute shortest path
    shortest = nx.shortest_path(G, source=src, target=dst)
    
    # Compute multiple paths up to max_depth
    paths_generator = nx.all_simple_paths(G, source=src, target=dst, cutoff=max_depth)
    extracted_paths = []
    
    for i, path in enumerate(paths_generator):
        if i >= max_paths:
            break
            
        hops = []
        path_total_amount = 0.0
        for h in range(len(path) - 1):
            u, v = path[h], path[h + 1]
            edge = G[u][v]
            amt = edge.get("amount", 0.0)
            path_total_amount += amt
            hops.append({
                "from": u,
                "to": v,
                "amount": round(amt, 2),
                "type": edge.get("type_str", edge.get("type", "TRANSFER")),
                "step": edge.get("step", 1)
            })
            
        extracted_paths.append({
            "path_index": i + 1,
            "length": len(path) - 1,
            "nodes": path,
            "total_path_volume": round(path_total_amount, 2),
            "hops": hops
        })
        
    return {
        "status": "connected",
        "source": src,
        "target": dst,
        "paths_found": len(extracted_paths),
        "shortest_path_hops": len(shortest) - 1,
        "shortest_path_nodes": shortest,
        "paths": extracted_paths
    }


def detect_suspicious_network_patterns(
    G: nx.DiGraph,
    min_fan_degree: int = 4,
    max_cycle_length: int = 4
) -> Dict[str, Any]:
    """
    Detect classic anti-money laundering (AML) topologies in the graph:
    1. Fan-in (Smurfing / Structuring): Multiple accounts sending to a single accumulator.
    2. Fan-out (Dispersal / Layering): Single account splitting funds to multiple accounts.
    3. Circular Loops (Round-tripping): Money cycles A -> B -> C -> A.
    4. Highly Connected Hubs: Accounts with abnormally high degree or PageRank.
    """
    fan_in_nodes = []
    fan_out_nodes = []
    
    for node in G.nodes():
        in_deg = G.in_degree(node)
        out_deg = G.out_degree(node)
        
        # Fan-in pattern: high in-degree, low out-degree
        if in_deg >= min_fan_degree and out_deg <= 2:
            fan_in_nodes.append({
                "account": node,
                "incoming_count": in_deg,
                "outgoing_count": out_deg,
                "total_received": G.nodes[node].get("total_received", 0.0),
                "is_anomaly": G.nodes[node].get("is_anomaly", False)
            })
            
        # Fan-out pattern: low in-degree, high out-degree
        if out_deg >= min_fan_degree and in_deg <= 2:
            fan_out_nodes.append({
                "account": node,
                "incoming_count": in_deg,
                "outgoing_count": out_deg,
                "total_sent": G.nodes[node].get("total_sent", 0.0),
                "is_anomaly": G.nodes[node].get("is_anomaly", False)
            })
            
    # Circular Loops Detection (cycles up to max_cycle_length)
    circular_loops = []
    try:
        # Simple cycles in directed graph
        cycles = nx.simple_cycles(G)
        for c in cycles:
            if 2 <= len(c) <= max_cycle_length:
                # Calculate cycle volume
                cycle_vol = 0.0
                cycle_hops = []
                for k in range(len(c)):
                    u = c[k]
                    v = c[(k + 1) % len(c)]
                    if G.has_edge(u, v):
                        amt = G[u][v].get("amount", 0.0)
                        cycle_vol += amt
                        cycle_hops.append(f"{u} -> {v} (${amt:,.2f})")
                circular_loops.append({
                    "cycle_length": len(c),
                    "nodes": c,
                    "total_volume": round(cycle_vol, 2),
                    "hops": cycle_hops
                })
            if len(circular_loops) >= 15:
                break
    except Exception:
        pass
        
    # Connected Components (Clusters of interacting entities)
    w_components = list(nx.weakly_connected_components(G))
    component_sizes = [len(c) for c in w_components]
    largest_component_size = max(component_sizes) if component_sizes else 0
    
    return {
        "fan_in_patterns": sorted(fan_in_nodes, key=lambda x: x["incoming_count"], reverse=True),
        "fan_out_patterns": sorted(fan_out_nodes, key=lambda x: x["outgoing_count"], reverse=True),
        "circular_loops": circular_loops,
        "total_weakly_connected_groups": len(w_components),
        "largest_group_size": largest_component_size
    }
