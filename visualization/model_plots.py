"""
visualization/model_plots.py
----------------------------
Visualization generators for Unsupervised Machine Learning models:
- Elbow Curve & Silhouette Plot for K-Means
- Dendrogram for Agglomerative Hierarchical Clustering
- Anomaly Score Distributions (Isolation Forest & LOF)
- 2D Scatter Visualizations for Clusters and Anomaly Highlights
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from scipy.cluster.hierarchy import dendrogram


# Configure dark-mode palette aesthetics
DARK_BG = "#0E1117"
DARK_CARD = "#1E232F"
TEXT_COLOR = "#FAFAFA"
ACCENT_CYAN = "#00F0FF"
ACCENT_RED = "#FF3366"
ACCENT_GREEN = "#00FF99"
ACCENT_AMBER = "#FFB800"
ACCENT_PURPLE = "#9D00FF"


def plot_elbow_curve(
    elbow_data: Dict[str, List[float]],
    title: str = "K-Means: Elbow Method & Silhouette Analysis"
) -> go.Figure:
    """
    Create a dual-axis interactive Plotly chart showing:
    1. Inertia (Elbow curve, Left axis)
    2. Silhouette Score (Right axis)
    """
    k_vals = elbow_data["k_values"]
    inertias = elbow_data["inertias"]
    silhouettes = elbow_data["silhouettes"]
    
    fig = go.Figure()
    
    # Inertia line
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=inertias,
        name="Inertia (WCSS)",
        mode="lines+markers",
        line=dict(color=ACCENT_CYAN, width=3),
        marker=dict(size=8, color=ACCENT_CYAN, symbol="circle")
    ))
    
    # Silhouette score line (secondary axis)
    fig.add_trace(go.Scatter(
        x=k_vals,
        y=silhouettes,
        name="Silhouette Score",
        mode="lines+markers",
        line=dict(color=ACCENT_GREEN, width=3, dash="dot"),
        marker=dict(size=8, color=ACCENT_GREEN, symbol="diamond"),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT_COLOR, size=16)),
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_CARD,
        xaxis=dict(
            title="Number of Clusters (K)",
            tickmode="linear",
            tick0=min(k_vals),
            dtick=1,
            gridcolor="#2D3748"
        ),
        yaxis=dict(
            title="Inertia (Within-Cluster Sum of Squares)",
            gridcolor="#2D3748",
            title_font=dict(color=ACCENT_CYAN),
            tickfont=dict(color=ACCENT_CYAN)
        ),
        yaxis2=dict(
            title="Silhouette Score",
            overlaying="y",
            side="right",
            range=[-0.1, 1.0],
            title_font=dict(color=ACCENT_GREEN),
            tickfont=dict(color=ACCENT_GREEN)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_COLOR)
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=450
    )
    
    return fig


def plot_dendrogram_matplotlib(
    linkage_matrix: np.ndarray,
    title: str = "Hierarchical Clustering Dendrogram",
    max_d: Optional[float] = None,
    p: int = 30
) -> plt.Figure:
    """
    Render a clean hierarchical clustering tree (dendrogram) using Matplotlib.
    p specifies the number of truncated branches for clarity.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=DARK_BG)
    ax.set_facecolor(DARK_CARD)
    
    dendrogram(
        linkage_matrix,
        truncate_mode="lastp",
        p=p,
        leaf_rotation=90,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax,
        color_threshold=max_d if max_d else 0.7 * max(linkage_matrix[:, 2])
    )
    
    ax.set_title(title, color=TEXT_COLOR, fontsize=14, pad=15)
    ax.set_xlabel("Account Cluster / Leaf Size", color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel("Ward Linkage Distance", color=TEXT_COLOR, fontsize=11)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color("#4A5568")
        
    if max_d:
        ax.axhline(y=max_d, c=ACCENT_RED, lw=1.5, linestyle="--", label=f"Threshold = {max_d:.2f}")
        ax.legend(facecolor=DARK_CARD, edgecolor="#4A5568", labelcolor=TEXT_COLOR)
        
    plt.tight_layout()
    return fig


def plot_anomaly_score_distribution(
    scores: np.ndarray,
    flags: np.ndarray,
    title: str = "Anomaly Score Distribution"
) -> go.Figure:
    """
    Plot interactive histogram and density of continuous anomaly scores.
    Separates inliers (Normal) from outliers (Flagged Anomalies).
    """
    df_plot = pd.DataFrame({
        "Score": scores,
        "Status": ["Anomaly Candidate" if f == -1 else "Normal Account" for f in flags]
    })
    
    fig = px.histogram(
        df_plot,
        x="Score",
        color="Status",
        color_discrete_map={
            "Normal Account": ACCENT_CYAN,
            "Anomaly Candidate": ACCENT_RED
        },
        barmode="overlay",
        nbins=50,
        opacity=0.75,
        title=title,
        template="plotly_dark"
    )
    
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_CARD,
        xaxis=dict(title="Normalized Anomaly Score (0.0 = Safe, 1.0 = Suspicious)", gridcolor="#2D3748"),
        yaxis=dict(title="Number of Accounts", gridcolor="#2D3748"),
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=40, t=50, b=40),
        height=400
    )
    
    return fig


def plot_clusters_2d_scatter(
    coords_2d: np.ndarray,
    labels: np.ndarray,
    account_ids: List[str],
    anomaly_flags: Optional[np.ndarray] = None,
    title: str = "2D Cluster Projection"
) -> go.Figure:
    """
    Interactive 2D scatter plot of account clusters with anomaly highlight rings.
    """
    status = []
    if anomaly_flags is not None:
        status = ["Anomaly Candidate" if f == -1 else "Normal" for f in anomaly_flags]
    else:
        status = ["Clustered" for _ in labels]
        
    df_viz = pd.DataFrame({
        "x": coords_2d[:, 0],
        "y": coords_2d[:, 1],
        "Cluster": [f"Cluster {l}" if l != -1 else "Noise (Outlier)" for l in labels],
        "Account": account_ids,
        "Status": status
    })
    
    fig = px.scatter(
        df_viz,
        x="x",
        y="y",
        color="Cluster",
        hover_data={"Account": True, "Status": True, "x": False, "y": False},
        title=title,
        template="plotly_dark",
        opacity=0.85
    )
    
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_CARD,
        xaxis=dict(title="Component 1", gridcolor="#2D3748"),
        yaxis=dict(title="Component 2", gridcolor="#2D3748"),
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=40, t=50, b=40),
        height=500
    )
    
    return fig
