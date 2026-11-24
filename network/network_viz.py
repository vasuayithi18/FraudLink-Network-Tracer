"""
network/network_viz.py
----------------------
Interactive Plotly network visualizations for financial transaction networks.

Key Features:
- Color-coded nodes: Cyan for Normal accounts, Bright Red/Amber for Anomaly Candidates,
  and Neon Green for the active investigated account.
- Dynamic node sizing based on transaction throughput or degree.
- Interactive hover tooltips displaying account ID, total volume sent/received, and anomaly score.
- Edge traces with transaction amount tooltips on hover.
- Ego-network extraction for deep-dive investigation into a specific suspect account.
"""

from typing import Dict, Any, List, Optional, Set
import numpy as np
import networkx as nx
import plotly.graph_objects as go

# Visual aesthetics (Dark cyber-financial palette)
DARK_BG = "#0E1117"
DARK_CARD = "#1A1F2C"
COLOR_NORMAL = "#00F0FF"      # Cyan for normal accounts
COLOR_ANOMALY = "#FF3366"     # Bright red for ML anomaly candidates
COLOR_FOCUS = "#00FF99"       # Neon green for investigated/searched account
COLOR_MERCHANT = "#B388FF"    # Purple for merchants
COLOR_EDGE = "rgba(100, 116, 139, 0.45)"
COLOR_ANOMALY_EDGE = "rgba(255, 51, 102, 0.7)"


def create_network_plotly(
    G: nx.DiGraph,
    focus_account: Optional[str] = None,
    path_nodes: Optional[List[str]] = None,
    max_nodes: int = 150,
    title: str = "Financial Transaction Network"
) -> go.Figure:
    """
    Generate an interactive 2D Plotly network graph.
    
    If the graph contains more than max_nodes, extracts a top-degree / anomaly-focused
    subgraph to ensure smooth rendering and readability.
    """
    if G.number_of_nodes() == 0:
        fig = go.Figure()
        fig.update_layout(
            title="Empty Transaction Network",
            template="plotly_dark",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_CARD
        )
        return fig
        
    # Subgraph extraction if graph is too dense
    if G.number_of_nodes() > max_nodes:
        # Prioritize focus account, path nodes, anomalies, and highest-degree nodes
        priority_nodes = set()
        if focus_account and G.has_node(focus_account):
            priority_nodes.add(focus_account)
            priority_nodes.update(G.neighbors(focus_account))
            priority_nodes.update(G.predecessors(focus_account))
            
        if path_nodes:
            priority_nodes.update([n for n in path_nodes if G.has_node(n)])
            
        # Add anomaly candidates
        for n, d in G.nodes(data=True):
            if d.get("is_anomaly", False):
                priority_nodes.add(n)
            if len(priority_nodes) >= max_nodes // 2:
                break
                
        # Fill remaining budget with highest degree nodes
        remaining_budget = max_nodes - len(priority_nodes)
        if remaining_budget > 0:
            top_deg = sorted(G.degree(), key=lambda x: x[1], reverse=True)
            for n, deg in top_deg:
                if n not in priority_nodes:
                    priority_nodes.add(n)
                if len(priority_nodes) >= max_nodes:
                    break
                    
        sub_G = G.subgraph(priority_nodes).copy()
    else:
        sub_G = G.copy()
        
    # 1. Compute 2D Node Layout using Spring / Fruchterman-Reingold algorithm
    pos = nx.spring_layout(sub_G, k=1.2 / np.sqrt(max(sub_G.number_of_nodes(), 1)), iterations=50, seed=42)
    
    # 2. Build Edge Traces & Edge Hover Labels
    edge_x = []
    edge_y = []
    edge_mid_x = []
    edge_mid_y = []
    edge_hover_text = []
    
    path_pairs = set()
    if path_nodes and len(path_nodes) > 1:
        for i in range(len(path_nodes) - 1):
            path_pairs.add((path_nodes[i], path_nodes[i + 1]))
            
    path_edge_x = []
    path_edge_y = []
    
    for u, v, d in sub_G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        
        # Check if edge is part of a highlighted traced path
        is_path_edge = (u, v) in path_pairs
        if is_path_edge:
            path_edge_x.extend([x0, x1, None])
            path_edge_y.extend([y0, y1, None])
        else:
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        # Compute midpoint for edge transaction amount hover tooltip
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        edge_mid_x.append(mx)
        edge_mid_y.append(my)
        
        amt = d.get("amount", 0.0)
        ttype = d.get("type_str", d.get("type", "TRANSFER"))
        count = d.get("count", 1)
        edge_hover_text.append(f"Transaction: {u} → {v}<br>Amount: ${amt:,.2f}<br>Type: {ttype}<br>Txn Count: {count}")

    # Standard edges trace
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.0, color=COLOR_EDGE),
        hoverinfo="none",
        mode="lines"
    )
    
    # Highlighted path edges trace (if path tracing active)
    path_edge_trace = go.Scatter(
        x=path_edge_x,
        y=path_edge_y,
        line=dict(width=3.5, color=COLOR_FOCUS),
        hoverinfo="none",
        mode="lines"
    )
    
    # Invisible markers at edge midpoints for transaction hover details
    edge_label_trace = go.Scatter(
        x=edge_mid_x,
        y=edge_mid_y,
        mode="markers",
        hoverinfo="text",
        text=edge_hover_text,
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        showlegend=False
    )
    
    # 3. Build Node Traces with Forensic Color & Sizing
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    node_outline_color = []
    node_outline_width = []
    
    for node in sub_G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        d = sub_G.nodes[node]
        is_anom = d.get("is_anomaly", False)
        is_focus = (node == focus_account)
        is_path_node = (path_nodes and node in path_nodes)
        is_merch = d.get("is_merchant", False)
        
        # Color coding logic
        if is_focus:
            c = COLOR_FOCUS
            outline_c = "#FFFFFF"
            outline_w = 3.0
            size = 22
        elif is_path_node:
            c = COLOR_FOCUS
            outline_c = COLOR_FOCUS
            outline_w = 2.0
            size = 18
        elif is_anom:
            c = COLOR_ANOMALY
            outline_c = "#FFAA00"
            outline_w = 2.0
            size = 16
        elif is_merch:
            c = COLOR_MERCHANT
            outline_c = "#4A5568"
            outline_w = 1.0
            size = 12
        else:
            c = COLOR_NORMAL
            outline_c = "#1E293B"
            outline_w = 1.0
            deg = sub_G.degree(node)
            size = max(9, min(24, 8 + int(np.log1p(deg) * 4)))
            
        node_color.append(c)
        node_size.append(size)
        node_outline_color.append(outline_c)
        node_outline_width.append(outline_w)
        
        # Detailed Hover Tooltip
        status_label = "<b>FOCUS INVESTIGATION</b>" if is_focus else ("⚠️ ANOMALY CANDIDATE" if is_anom else "NORMAL ACCOUNT")
        hover_info = (
            f"<b>Account:</b> {node}<br>"
            f"<b>Status:</b> {status_label}<br>"
            f"<b>Anomaly Score:</b> {d.get('anomaly_score', 0.0):.3f}<br>"
            f"<b>Total Sent:</b> ${d.get('total_sent', 0.0):,.2f}<br>"
            f"<b>Total Received:</b> ${d.get('total_received', 0.0):,.2f}<br>"
            f"<b>Net Flow:</b> ${d.get('net_flow', 0.0):,.2f}<br>"
            f"<b>In-Degree (Incoming):</b> {d.get('in_degree', 0)}<br>"
            f"<b>Out-Degree (Outgoing):</b> {d.get('out_degree', 0)}"
        )
        node_text.append(hover_info)
        
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text" if sub_G.number_of_nodes() <= 35 else "markers",
        text=[n if (n == focus_account or is_anom or sub_G.number_of_nodes() <= 20) else "" for n in sub_G.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(color=node_outline_color, width=node_outline_width)
        )
    )
    
    fig = go.Figure(data=[edge_trace, path_edge_trace, edge_label_trace, node_trace])
    
    fig.update_layout(
        title=dict(
            text=f"{title} ({sub_G.number_of_nodes()} accounts, {sub_G.number_of_edges()} transactions)",
            font=dict(color="#FAFAFA", size=15)
        ),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_CARD,
        height=600
    )
    
    return fig


def create_ego_network_plotly(
    G: nx.DiGraph,
    center_account: str,
    radius: int = 1
) -> go.Figure:
    """
    Generate an Ego-Network centered on a single target account.
    Extracts the direct 1-hop or 2-hop neighborhood of the target account.
    Ideal for in-depth investigation of flagged suspects.
    """
    acc = str(center_account).strip()
    if not G.has_node(acc):
        fig = go.Figure()
        fig.update_layout(
            title=f"Account '{acc}' not found in network",
            template="plotly_dark",
            paper_bgcolor=DARK_BG
        )
        return fig
        
    ego_nodes = {acc}
    current_frontier = {acc}
    for _ in range(radius):
        next_frontier = set()
        for n in current_frontier:
            next_frontier.update(G.predecessors(n))
            next_frontier.update(G.successors(n))
        ego_nodes.update(next_frontier)
        current_frontier = next_frontier
        
    sub_G = G.subgraph(ego_nodes).copy()
    return create_network_plotly(
        sub_G,
        focus_account=acc,
        max_nodes=100,
        title=f"Ego Network for Suspect Account: {acc}"
    )
