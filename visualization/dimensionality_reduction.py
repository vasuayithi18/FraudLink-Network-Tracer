"""
visualization/dimensionality_reduction.py
-----------------------------------------
Dimensionality reduction algorithms (PCA and t-SNE) for projecting high-dimensional
account behavioral features (18 dimensions) into human-interpretable 2D spaces.

Viva Concepts:
- PCA (Principal Component Analysis):
  Linear projection technique that finds orthogonal directions (principal components)
  maximizing variance. Preserves global geometric data structure. Fast O(N * D * min(N, D)).
- t-SNE (t-Distributed Stochastic Neighbor Embedding):
  Non-linear manifold technique that minimizes Kullback-Leibler divergence between
  neighborhood probability distributions in high-dim vs low-dim. Excels at preserving
  local neighborhood clusters. Subsampled for real-time responsiveness.
"""

from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DARK_BG = "#0E1117"
DARK_CARD = "#1A1F2C"
COLOR_NORMAL = "#00F0FF"      # Cyan
COLOR_ANOMALY = "#FF3366"     # Red
COLOR_MERCHANT = "#B388FF"


def compute_pca(
    X: np.ndarray,
    n_components: int = 2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis.
    
    Returns:
    --------
    dict with:
    - coords: 2D projection (N, 2)
    - explained_variance: array of variance ratio explained by each component
    - total_variance_explained: sum of explained variance
    - components: principal component axes
    - model: fitted PCA instance
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(X)
    
    return {
        "coords": coords,
        "explained_variance": pca.explained_variance_ratio_,
        "total_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
        "components": pca.components_,
        "model": pca
    }


def compute_tsne(
    X: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    max_samples: int = 2000,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Perform t-SNE manifold learning on account feature space.
    Subsamples if N > max_samples to guarantee sub-2-second execution.
    """
    N = len(X)
    if N > max_samples:
        np.random.seed(random_state)
        sample_indices = np.random.choice(N, size=max_samples, replace=False)
        X_sub = X[sample_indices]
        sampled = True
    else:
        sample_indices = np.arange(N)
        X_sub = X
        sampled = False
        
    # Cap perplexity to safe fraction of sample size
    safe_perp = min(perplexity, max(5.0, (len(X_sub) - 1) / 3.5))
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=safe_perp,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=random_state
    )
    coords_sub = tsne.fit_transform(X_sub)
    
    return {
        "coords": coords_sub,
        "sample_indices": sample_indices,
        "sampled": sampled,
        "perplexity": safe_perp,
        "kl_divergence": float(tsne.kl_divergence_) if hasattr(tsne, "kl_divergence_") else None
    }


def plot_pca_vs_tsne_side_by_side(
    pca_data: Dict[str, Any],
    tsne_data: Dict[str, Any],
    account_ids: List[str],
    labels: np.ndarray,
    anomaly_flags: np.ndarray
) -> go.Figure:
    """
    Create a dual-panel side-by-side interactive Plotly figure comparing
    PCA (Global Variance) vs t-SNE (Local Manifold Clusters).
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"PCA Projection (Explained Variance: {pca_data['total_variance_explained']*100:.1f}%)",
            f"t-SNE Projection (Perplexity = {tsne_data['perplexity']:.0f})"
        ),
        horizontal_spacing=0.08
    )
    
    # 1. PCA Panel (Left)
    pca_coords = pca_data["coords"]
    status_pca = ["Anomaly Candidate" if f == -1 else "Normal Account" for f in anomaly_flags]
    colors_pca = [COLOR_ANOMALY if f == -1 else COLOR_NORMAL for f in anomaly_flags]
    
    fig.add_trace(
        go.Scatter(
            x=pca_coords[:, 0],
            y=pca_coords[:, 1],
            mode="markers",
            marker=dict(size=6, color=colors_pca, opacity=0.8),
            text=[f"Account: {acc}<br>Cluster: {lbl}<br>Status: {st}" for acc, lbl, st in zip(account_ids, labels, status_pca)],
            hoverinfo="text",
            name="PCA Accounts",
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. t-SNE Panel (Right)
    tsne_coords = tsne_data["coords"]
    sub_idx = tsne_data["sample_indices"]
    sub_accounts = [account_ids[i] for i in sub_idx]
    sub_labels = labels[sub_idx]
    sub_flags = anomaly_flags[sub_idx]
    status_tsne = ["Anomaly Candidate" if f == -1 else "Normal Account" for f in sub_flags]
    colors_tsne = [COLOR_ANOMALY if f == -1 else COLOR_NORMAL for f in sub_flags]
    
    fig.add_trace(
        go.Scatter(
            x=tsne_coords[:, 0],
            y=tsne_coords[:, 1],
            mode="markers",
            marker=dict(size=6, color=colors_tsne, opacity=0.8),
            text=[f"Account: {acc}<br>Cluster: {lbl}<br>Status: {st}" for acc, lbl, st in zip(sub_accounts, sub_labels, status_tsne)],
            hoverinfo="text",
            name="t-SNE Accounts",
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_CARD,
        font=dict(color="#FAFAFA"),
        height=520,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.update_xaxes(title_text="Principal Component 1", gridcolor="#2D3748", row=1, col=1)
    fig.update_yaxes(title_text="Principal Component 2", gridcolor="#2D3748", row=1, col=1)
    fig.update_xaxes(title_text="t-SNE Dimension 1", gridcolor="#2D3748", row=1, col=2)
    fig.update_yaxes(title_text="t-SNE Dimension 2", gridcolor="#2D3748", row=1, col=2)
    
    return fig
