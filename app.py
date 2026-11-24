"""
app.py
------
FraudLink + Network Tracer
Subject: INT396 – Unsupervised Learning

A Unified Intelligent Financial Transaction Analysis & Network Investigation System.
Integrates Unsupervised Machine Learning (K-Means, DBSCAN, Hierarchical, Isolation Forest, LOF),
Graph Intelligence (NetworkX DiGraph, Multi-hop Path Tracing, AML Topology Mining),
Dimensionality Reduction (PCA vs t-SNE), and Forensic Investigation Reports.
"""

import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Preprocessing modules
from preprocessing.loader import load_dataset, get_dataset_summary, clean_dataset
from preprocessing.feature_engineering import extract_account_features, get_feature_names
from preprocessing.scaler import scale_features

# Machine learning modules
from models.clustering import compute_elbow_curve, run_kmeans, run_dbscan, run_hierarchical, compute_hierarchical_linkage
from models.anomaly import run_isolation_forest, run_lof
from models.evaluation import evaluate_clusters, compare_all_models

# Network modules
from network.graph_builder import build_transaction_graph, get_graph_summary
from network.tracer import get_account_investigation, trace_transaction_paths, detect_suspicious_network_patterns
from network.network_viz import create_network_plotly, create_ego_network_plotly

# Visualization & Reporting modules
from visualization.model_plots import (
    plot_elbow_curve, plot_dendrogram_matplotlib,
    plot_anomaly_score_distribution, plot_clusters_2d_scatter
)
from visualization.dimensionality_reduction import (
    compute_pca, compute_tsne, plot_pca_vs_tsne_side_by_side
)
from reports.reporter import generate_forensic_investigation_report, export_report_to_csv


# ==============================================================================
# STREAMLIT PAGE CONFIG & CUSTOM DARK THEME CSS
# ==============================================================================
st.set_page_config(
    page_title="FraudLink + Network Tracer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyber-Financial Dark Theme Styling
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #0E1117;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #131722;
        border-right: 1px solid #1E293B;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1E232F;
        border: 1px solid #2D3748;
        border-radius: 10px;
        padding: 12px 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"] label {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #00F0FF !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }
    
    /* Header Badges */
    .hero-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(0, 240, 255, 0.15);
        color: #00F0FF;
        border: 1px solid rgba(0, 240, 255, 0.4);
        margin-bottom: 8px;
    }
    
    .risk-critical {
        background-color: rgba(255, 51, 102, 0.2);
        color: #FF3366;
        border: 1px solid #FF3366;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .risk-high {
        background-color: rgba(255, 184, 0, 0.2);
        color: #FFB800;
        border: 1px solid #FFB800;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    
    .risk-medium {
        background-color: rgba(0, 240, 255, 0.2);
        color: #00F0FF;
        border: 1px solid #00F0FF;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: 600;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1E232F;
        border-radius: 6px;
        color: #94A3B8;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00F0FF !important;
        color: #0E1117 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CACHED DATA PIPELINE & SESSION STATE INITIALIZATION
# ==============================================================================
@st.cache_data(show_spinner="Loading and sampling dataset...")
def get_cached_raw_data(source_choice: str, uploaded_bytes=None, sample_size=5000):
    """Load transactions safely with caching."""
    if source_choice == "Custom Upload" and uploaded_bytes is not None:
        buffer = io.BytesIO(uploaded_bytes)
        df = load_dataset(buffer, sample_size=sample_size)
    elif source_choice == "Full Transactions (transactions.csv)" and os.path.exists("data/transactions.csv"):
        df = load_dataset("data/transactions.csv", sample_size=sample_size)
    else:
        sample_path = "data/paysim_sample.csv"
        if not os.path.exists(sample_path) and os.path.exists("data/transactions.csv"):
            sample_path = "data/transactions.csv"
        df = load_dataset(sample_path, sample_size=sample_size)
    df_clean, audit = clean_dataset(df)
    return df_clean, audit


@st.cache_data(show_spinner="Extracting account behavioral profiles...")
def get_cached_account_features(df: pd.DataFrame):
    """Extract account features and standardize."""
    features_df, ground_truth = extract_account_features(df)
    feature_cols = get_feature_names()
    scaled_df, scaler = scale_features(features_df, feature_cols)
    return features_df, scaled_df, ground_truth, feature_cols


@st.cache_data(show_spinner="Running unsupervised ML models...")
def get_cached_ml_results(scaled_values: np.ndarray, k=4, eps=1.5, min_samples=5, contamination=0.03):
    """Run all 5 unsupervised algorithms."""
    results = {}
    results["K-Means"] = run_kmeans(scaled_values, n_clusters=k)
    results["DBSCAN"] = run_dbscan(scaled_values, eps=eps, min_samples=min_samples)
    results["Hierarchical"] = run_hierarchical(scaled_values, n_clusters=k, sample_limit=2000)
    results["Isolation Forest"] = run_isolation_forest(scaled_values, contamination=contamination)
    results["Local Outlier Factor"] = run_lof(scaled_values, contamination=contamination)
    return results


# ==============================================================================
# SIDEBAR NAVIGATION & DATASET CONTROLS
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="hero-badge">INT396 Project</div>', unsafe_allow_html=True)
    st.markdown("### 🛡️ FraudLink\n**Network Tracer v1.0**")
    st.caption("Unsupervised Learning & Graph Intelligence")
    st.markdown("---")
    
    # Dataset Selector
    st.subheader("📁 Data Source")
    available_sources = ["PaySim Starter Sample (4,791 txns)"]
    if os.path.exists("data/transactions.csv"):
        available_sources.append("Full Transactions (transactions.csv)")
    available_sources.append("Custom Upload")
    
    source_choice = st.selectbox("Select Dataset", available_sources, index=0)
    
    uploaded_bytes = None
    if source_choice == "Custom Upload":
        uploaded_file = st.file_uploader("Upload Transaction CSV", type=["csv"])
        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            
    sample_size = st.slider(
        "Analysis Sample Size",
        min_value=1000,
        max_value=25000,
        value=5000,
        step=1000,
        help="Manageable sample ensures sub-second clustering & smooth graph rendering."
    )
    
    st.markdown("---")
    
    # Navigation Switcher
    st.subheader("🧭 Navigation")
    pages = [
        "1. Overview Dashboard",
        "2. Dataset Explorer",
        "3. Data Preprocessing",
        "4. ML Analysis",
        "5. Network Tracer",
        "6. Anomaly Detection",
        "7. PCA vs t-SNE",
        "8. Reports"
    ]
    selected_page = st.radio("Go to Page", pages, index=0)
    st.markdown("---")
    st.caption("🚀 Running in `.venv` (Python 3.11)")


# Load Data and Precompute State
df_tx, clean_audit = get_cached_raw_data(source_choice, uploaded_bytes, sample_size)
features_df, scaled_df, ground_truth, feature_cols = get_cached_account_features(df_tx)
ml_results = get_cached_ml_results(scaled_df.values)

# Extract identified anomalies (Isolation Forest + Consensus)
iso_flags = ml_results["Isolation Forest"]["flags"]
anomaly_candidate_ids = set(scaled_df.index[iso_flags == -1])
anomaly_scores_map = dict(zip(scaled_df.index, ml_results["Isolation Forest"]["scores"]))


# ==============================================================================
# PAGE 1: OVERVIEW DASHBOARD
# ==============================================================================
if selected_page == "1. Overview Dashboard":
    st.markdown('<div class="hero-badge">Module 1 • System Command Center</div>', unsafe_allow_html=True)
    st.title("Financial Fraud Intelligence Dashboard")
    st.markdown("Automated unsupervised financial transaction surveillance, entity profiling, and network tracking.")
    
    st.info(
        "⚖️ **Forensic Compliance Notice:** Flagged accounts are *unsupervised anomaly candidates* "
        "identified based on statistical deviations and topological graph patterns. They represent priority candidates "
        "for manual compliance investigation, not confirmed fraudulent entities. Monetary values are denominated in local currency units (as specified in PaySim documentation)."
    )
    
    summary = get_dataset_summary(df_tx)
    
    # Top 4 KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Transactions", f"{summary['total_rows']:,}", delta=f"{summary['total_cols']} Columns")
    with kpi2:
        st.metric("Unique Accounts", f"{summary['total_unique_accounts']:,}", delta=f"{summary['unique_senders']:,} Senders")
    with kpi3:
        st.metric("Financial Volume", f"{summary['total_volume']:,.2f} Units", delta=f"Avg {summary['avg_amount']:,.2f} Units")
    with kpi4:
        st.metric("Anomaly Candidates", f"{len(anomaly_candidate_ids):,}", delta=f"{(len(anomaly_candidate_ids)/len(features_df))*100:.1f}% Outlier Rate")
        
    st.markdown("###")
    
    # Charts Row
    col_chart1, col_chart2 = st.columns([1, 1])
    with col_chart1:
        st.subheader("📊 Transaction Types Distribution")
        type_df = pd.DataFrame(list(summary["transaction_types"].items()), columns=["Type", "Count"])
        fig_type = px.pie(
            type_df, values="Count", names="Type",
            hole=0.45,
            color_discrete_sequence=["#00F0FF", "#38BDF8", "#818CF8", "#A78BFA", "#F472B6"],
            template="plotly_dark"
        )
        fig_type.update_layout(paper_bgcolor="#1E232F", height=350, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_type, width="stretch")
        
    with col_chart2:
        st.subheader("📈 Anomaly Score Distribution")
        fig_scores = plot_anomaly_score_distribution(
            ml_results["Isolation Forest"]["scores"],
            ml_results["Isolation Forest"]["flags"],
            title="Account Anomaly Score Frequencies"
        )
        st.plotly_chart(fig_scores, width="stretch")
        
    st.markdown("---")
    
    # Recent Flagged Accounts Table
    st.subheader("🚨 Priority Suspicious Account Candidates")
    top_suspects_df = generate_forensic_investigation_report(features_df, ml_results, top_n=8)
    st.dataframe(top_suspects_df, width="stretch")


# ==============================================================================
# PAGE 2: DATASET EXPLORER
# ==============================================================================
elif selected_page == "2. Dataset Explorer":
    st.markdown('<div class="hero-badge">Module 2 • Dataset Inspection & Audit</div>', unsafe_allow_html=True)
    st.title("Financial Dataset Explorer")
    st.markdown("Inspect raw transaction logs, schema specifications, missing value audits, and summary statistics.")
    
    summary = get_dataset_summary(df_tx)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows Loaded", f"{summary['total_rows']:,}")
    m2.metric("Total Attributes", summary["total_cols"])
    m3.metric("Memory Footprint", f"{summary['memory_mb']} MB")
    m4.metric("Duplicates Removed", clean_audit["duplicates_removed"])
    
    st.markdown("###")
    
    # Sub-tabs for Explorer
    tab_data, tab_schema, tab_stats, tab_missing = st.tabs([
        "📄 Raw Records", "🔍 Schema & Data Types", "📊 Statistical Summary", "⚠️ Missing & Quality Audit"
    ])
    
    with tab_data:
        st.subheader("Transaction Records Preview")
        st.dataframe(df_tx.head(100), width="stretch")
        
    with tab_schema:
        st.subheader("Column Metadata & Data Types")
        schema_df = pd.DataFrame({
            "Column Name": df_tx.columns,
            "Data Type": [str(t) for t in df_tx.dtypes],
            "Non-Null Count": df_tx.count().values,
            "Null Count": df_tx.isnull().sum().values,
            "Unique Values": [df_tx[c].nunique() for c in df_tx.columns]
        })
        st.dataframe(schema_df, width="stretch")
        
    with tab_stats:
        st.subheader("Numerical Attributes Statistics")
        st.dataframe(df_tx.describe().T, width="stretch")
        
    with tab_missing:
        st.subheader("Missing Values & Data Quality Audit")
        st.dataframe(summary["missing_summary"], width="stretch")
        if clean_audit["missing_handled"] > 0:
            st.info(f"Handled {clean_audit['missing_handled']} missing values during ingestion.")
        else:
            st.success("✓ Zero missing cells detected in transaction dataset.")


# ==============================================================================
# PAGE 3: DATA PREPROCESSING
# ==============================================================================
elif selected_page == "3. Data Preprocessing":
    st.markdown('<div class="hero-badge">Module 3 • Feature Engineering & Standardization</div>', unsafe_allow_html=True)
    st.title("Entity Profiling & Feature Engineering")
    st.markdown(
        "Transforming raw transaction streams into rich **Account Behavioral Profiles** for unsupervised clustering. "
        "Distance-based algorithms require zero mean and unit variance normalization."
    )
    
    col_desc, col_counts = st.columns([2, 1])
    with col_desc:
        st.info(
            "**Forensic Feature Engineering Rationale:**\n\n"
            "- Individual transactions only tell an instantaneous story. Money laundering networks, structuring, "
            "and mule rings operate across **accounts over time**.\n"
            "- Extracted 18 behavioral indicators including Net Flow ($Received - Sent$), Outflow Ratios, "
            "Counterparty Degree, Balance Drains, and High-Risk Transaction Frequencies."
        )
    with col_counts:
        st.metric("Total Account Profiles", f"{len(features_df):,}")
        st.metric("Extracted Features", f"{len(feature_cols)} Dimensions")
        
    st.markdown("###")
    
    tab_feat_table, tab_scaling, tab_corr = st.tabs([
        "📋 Engineered Feature Registry", "⚖️ StandardScaler Verification", "🔥 Feature Correlation Matrix"
    ])
    
    with tab_feat_table:
        st.subheader("Extracted Account Feature Matrix (Unscaled)")
        st.dataframe(features_df.head(50), width="stretch")
        
    with tab_scaling:
        st.subheader("StandardScaler Effect (Zero Mean & Unit Variance)")
        selected_feature = st.selectbox("Select Feature to Compare", feature_cols, index=0)
        
        col_raw, col_scaled = st.columns(2)
        with col_raw:
            st.markdown(f"**Before Scaling:** `{selected_feature}`")
            fig_raw = px.histogram(features_df, x=selected_feature, nbins=40, template="plotly_dark", color_discrete_sequence=["#94A3B8"])
            fig_raw.update_layout(paper_bgcolor="#1E232F", plot_bgcolor="#131722", height=320)
            st.plotly_chart(fig_raw, width="stretch")
            st.caption(f"Mean: {features_df[selected_feature].mean():,.2f} | Std: {features_df[selected_feature].std():,.2f}")
            
        with col_scaled:
            st.markdown(f"**After StandardScaler:** `Z-score({selected_feature})`")
            fig_scaled = px.histogram(scaled_df, x=selected_feature, nbins=40, template="plotly_dark", color_discrete_sequence=["#00F0FF"])
            fig_scaled.update_layout(paper_bgcolor="#1E232F", plot_bgcolor="#131722", height=320)
            st.plotly_chart(fig_scaled, width="stretch")
            st.caption(f"Mean: {scaled_df[selected_feature].mean():.4f} (≈0.0) | Std: {scaled_df[selected_feature].std():.4f} (≈1.0)")
            
    with tab_corr:
        st.subheader("Behavioral Feature Correlation Heatmap")
        corr_matrix = scaled_df[feature_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_corr.update_layout(paper_bgcolor="#1E232F", height=600)
        st.plotly_chart(fig_corr, width="stretch")


# ==============================================================================
# PAGE 4: MACHINE LEARNING ANALYSIS
# ==============================================================================
elif selected_page == "4. ML Analysis":
    st.markdown('<div class="hero-badge">Module 4 • Unsupervised Clustering & Evaluation</div>', unsafe_allow_html=True)
    st.title("Unsupervised Machine Learning Models")
    st.markdown("Benchmarking 5 clustering and anomaly detection algorithms on account behavioral patterns.")
    
    # Algorithm Sub-Tabs
    tab_comp, tab_km, tab_db, tab_hier, tab_iso, tab_lof = st.tabs([
        "🏆 Model Comparison", "1️⃣ K-Means", "2️⃣ DBSCAN", "3️⃣ Hierarchical", "4️⃣ Isolation Forest", "5️⃣ LOF"
    ])
    
    with tab_comp:
        st.subheader("Master Algorithm Benchmarking Matrix")
        comp_df = compare_all_models(scaled_df.values, ml_results, ground_truth=ground_truth)
        st.dataframe(comp_df, width="stretch")
        
        st.markdown("###")
        col_sil, col_dbi = st.columns(2)
        with col_sil:
            st.subheader("Silhouette Scores (Higher is Better)")
            valid_sil = comp_df[comp_df["Silhouette Score"] != "N/A"].copy()
            valid_sil["Silhouette Score"] = valid_sil["Silhouette Score"].astype(float)
            fig_sil = px.bar(
                valid_sil, x="Algorithm", y="Silhouette Score",
                color="Algorithm", template="plotly_dark",
                color_discrete_sequence=["#00F0FF", "#38BDF8", "#818CF8", "#A78BFA", "#F472B6"]
            )
            fig_sil.update_layout(paper_bgcolor="#1E232F", plot_bgcolor="#131722", showlegend=False, height=350)
            st.plotly_chart(fig_sil, width="stretch")
            
        with col_dbi:
            st.subheader("Davies-Bouldin Index (Lower is Better)")
            valid_dbi = comp_df[comp_df["Davies-Bouldin"] != "N/A"].copy()
            valid_dbi["Davies-Bouldin"] = valid_dbi["Davies-Bouldin"].astype(float)
            fig_dbi = px.bar(
                valid_dbi, x="Algorithm", y="Davies-Bouldin",
                color="Algorithm", template="plotly_dark",
                color_discrete_sequence=["#F43F5E", "#FB7185", "#FDA4AF", "#FDE047", "#A3E635"]
            )
            fig_dbi.update_layout(paper_bgcolor="#1E232F", plot_bgcolor="#131722", showlegend=False, height=350)
            st.plotly_chart(fig_dbi, width="stretch")
            
    with tab_km:
        st.subheader("K-Means Clustering & Elbow Method")
        k_val = st.slider("Select K (Clusters)", min_value=2, max_value=8, value=4, key="km_k_slider")
        km_res = run_kmeans(scaled_df.values, n_clusters=k_val)
        
        elbow_data = compute_elbow_curve(scaled_df.values, k_range=range(2, 7))
        fig_elbow = plot_elbow_curve(elbow_data)
        st.plotly_chart(fig_elbow, width="stretch")
        
        st.markdown(f"**Fitted K={k_val}:** Inertia = `{km_res['inertia']:,.2f}` | Outliers (Minority Clusters) = `{np.sum(km_res['anomaly_flags'] == -1)}`")
        
    with tab_db:
        st.subheader("DBSCAN Density-Based Clustering")
        c1, c2 = st.columns(2)
        eps_val = c1.slider("Epsilon (eps)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
        min_samp = c2.slider("Min Samples (min_samples)", min_value=3, max_value=15, value=5, step=1)
        
        db_res = run_dbscan(scaled_df.values, eps=eps_val, min_samples=min_samp)
        
        d1, d2, d3 = st.columns(3)
        d1.metric("Clusters Discovered", db_res["num_clusters"])
        d2.metric("Noise Outliers Flagged", f"{db_res['noise_count']} accounts")
        d3.metric("Noise Percentage", f"{db_res['noise_pct']}%")
        
        pt_df = pd.DataFrame(list(pd.Series(db_res["point_types"]).value_counts().items()), columns=["Type", "Count"])
        fig_pts = px.pie(pt_df, values="Count", names="Type", hole=0.4, template="plotly_dark", color_discrete_sequence=["#00F0FF", "#FFB800", "#FF3366"])
        fig_pts.update_layout(paper_bgcolor="#1E232F", height=320)
        st.plotly_chart(fig_pts, width="stretch")
        
    with tab_hier:
        st.subheader("Hierarchical Agglomerative Clustering & Dendrogram")
        hcol1, hcol2 = st.columns(2)
        k_hier = hcol1.slider("Number of Clusters (K)", min_value=2, max_value=8, value=4, key="hier_k_slider")
        link_method = hcol2.selectbox("Linkage Criterion", ["ward", "complete", "average"], index=0)
        
        hier_res = run_hierarchical(scaled_df.values, n_clusters=k_hier, linkage_method=link_method, sample_limit=2000)
        st.markdown(f"**Hierarchical Fit (K={k_hier}, Linkage={link_method}):** Cluster counts: `{hier_res['cluster_counts']}` | Minority Outliers: `{np.sum(hier_res['anomaly_flags'] == -1)}`")
        
        linkage_mat = compute_hierarchical_linkage(scaled_df.values, method=link_method, sample_size=100)
        fig_dendro = plot_dendrogram_matplotlib(linkage_mat, max_d=25.0)
        st.pyplot(fig_dendro)
        
    with tab_iso:
        st.subheader("Isolation Forest Anomaly Isolation")
        iso_contam = st.slider("Contamination Rate (Expected Outliers %)", min_value=0.01, max_value=0.10, value=0.03, step=0.01, key="iso_contam_slider")
        iso_res = run_isolation_forest(scaled_df.values, contamination=iso_contam)
        st.metric("Total Anomaly Candidates Flagged", f"{iso_res['outlier_count']} accounts", delta=f"{iso_res['outlier_pct']}% of population")
        fig_iso = plot_anomaly_score_distribution(iso_res["scores"], iso_res["flags"], title=f"Isolation Forest Score Spectrum (Contamination={iso_contam:.2f})")
        st.plotly_chart(fig_iso, width="stretch")
        
    with tab_lof:
        st.subheader("Local Outlier Factor (LOF)")
        lcol1, lcol2 = st.columns(2)
        lof_n = lcol1.slider("Number of Neighbors (k)", min_value=10, max_value=50, value=20, step=5, key="lof_n_slider")
        lof_contam = lcol2.slider("Contamination Rate", min_value=0.01, max_value=0.10, value=0.03, step=0.01, key="lof_contam_slider")
        lof_res = run_lof(scaled_df.values, n_neighbors=lof_n, contamination=lof_contam)
        st.metric("Total Local Density Outliers", f"{lof_res['outlier_count']} accounts", delta=f"{lof_res['outlier_pct']}% of population")
        fig_lof = plot_anomaly_score_distribution(lof_res["scores"], lof_res["flags"], title=f"Local Outlier Factor Score Spectrum (k={lof_n}, Contamination={lof_contam:.2f})")
        st.plotly_chart(fig_lof, width="stretch")


# ==============================================================================
# PAGE 5: NETWORK TRACER
# ==============================================================================
elif selected_page == "5. Network Tracer":
    st.markdown('<div class="hero-badge">Module 6 • Graph Intelligence & Forensics</div>', unsafe_allow_html=True)
    st.title("Transaction Network Tracer")
    st.markdown(
        "Explore directed transaction graphs using NetworkX and interactive Plotly visualization. "
        "Search accounts, inspect counterparty relationships, trace multi-hop money laundering paths, and detect AML topologies."
    )
    
    # Graph Construction Controls
    with st.expander("⚙️ Graph Construction & Filtering Controls", expanded=True):
        fcol1, fcol2, fcol3 = st.columns(3)
        min_txn_amt = fcol1.slider("Minimum Amount Filter ($)", min_value=0, max_value=25000, value=250, step=250)
        all_types = list(df_tx["type"].unique()) if "type" in df_tx.columns else ["TRANSFER"]
        selected_types = fcol2.multiselect("Transaction Types", all_types, default=all_types)
        max_edges_render = fcol3.slider("Max Graph Edges", min_value=200, max_value=3000, value=1200, step=200)
        
    # Build Directed Graph
    G = build_transaction_graph(
        df_tx,
        anomaly_candidates=anomaly_candidate_ids,
        anomaly_scores=anomaly_scores_map,
        min_amount=float(min_txn_amt),
        txn_types=selected_types,
        max_edges=max_edges_render
    )
    g_summary = get_graph_summary(G)
    
    # Graph Metrics Row
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Graph Nodes (Accounts)", f"{g_summary['num_nodes']:,}")
    g2.metric("Directed Edges", f"{g_summary['num_edges']:,}")
    g3.metric("Network Volume", f"${g_summary['total_volume']:,.2f}")
    g4.metric("Flagged Anomaly Nodes", f"{g_summary['anomaly_nodes_count']} ({g_summary['anomaly_pct']}%)")
    
    st.markdown("###")
    
    net_tab_viz, net_tab_search, net_tab_path, net_tab_topo = st.tabs([
        "🌐 Interactive Network Graph", "🔍 Account Search & Ego-Network", "🛤️ Multi-Hop Path Tracer", "🕸️ AML Topologies"
    ])
    
    with net_tab_viz:
        st.subheader("Global Transaction Flow Visualization")
        fig_global = create_network_plotly(G, max_nodes=150, title="Directed Transaction Network Graph")
        st.plotly_chart(fig_global, width="stretch")
        
    with net_tab_search:
        st.subheader("Account Deep-Dive Investigation")
        
        # Pick top suspect or custom input
        top_suspect_candidate = list(anomaly_candidate_ids)[0] if anomaly_candidate_ids else list(G.nodes())[0]
        searched_account = st.text_input("Enter Account ID to Investigate:", value=top_suspect_candidate)
        
        investigation = get_account_investigation(G, searched_account)
        if investigation:
            # Account Profile Card
            inv1, inv2, inv3, inv4 = st.columns(4)
            inv1.metric("Status", "⚠️ ANOMALY" if investigation["is_anomaly"] else "NORMAL")
            inv2.metric("Total Sent", f"${investigation['total_sent']:,.2f}", f"{investigation['out_degree']} txns")
            inv3.metric("Total Received", f"${investigation['total_received']:,.2f}", f"{investigation['in_degree']} txns")
            inv4.metric("Net Flow", f"${investigation['net_flow']:,.2f}")
            
            # Ego-Network Visualization
            st.subheader(f"Ego-Network: Direct Counterparties of {searched_account}")
            fig_ego = create_ego_network_plotly(G, searched_account, radius=1)
            st.plotly_chart(fig_ego, width="stretch")
            
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("**Incoming Transactions (Predecessors):**")
                if investigation["incoming_transactions"]:
                    st.dataframe(pd.DataFrame(investigation["incoming_transactions"]), width="stretch")
                else:
                    st.caption("No incoming transactions in active graph.")
                    
            with col_out:
                st.markdown("**Outgoing Transactions (Successors):**")
                if investigation["outgoing_transactions"]:
                    st.dataframe(pd.DataFrame(investigation["outgoing_transactions"]), width="stretch")
                else:
                    st.caption("No outgoing transactions in active graph.")
        else:
            st.warning(f"Account '{searched_account}' was not found in the currently filtered graph.")
            
    with net_tab_path:
        st.subheader("Multi-Hop Transaction Path Tracing")
        st.caption("Trace money laundering chains and intermediary mule accounts between two entities.")
        
        pcol1, pcol2 = st.columns(2)
        default_src = "C6004378" if G.has_node("C6004378") else list(G.nodes())[0]
        default_dst = "C8004804" if G.has_node("C8004804") else list(G.nodes())[-1]
        
        src_input = pcol1.text_input("Source Account (Sender):", value=default_src)
        dst_input = pcol2.text_input("Destination Account (Receiver):", value=default_dst)
        
        if st.button("Trace Transaction Trajectory", type="primary"):
            path_res = trace_transaction_paths(G, src_input, dst_input, max_paths=3, max_depth=5)
            if path_res["status"] == "connected":
                st.success(f"Path Discovered! Length: {path_res['shortest_path_hops']} hop(s)")
                st.markdown(f"**Shortest Path:** `{' -> '.join(path_res['shortest_path_nodes'])}`")
                
                # Render network focused on path
                fig_path = create_network_plotly(
                    G,
                    focus_account=src_input,
                    path_nodes=path_res["shortest_path_nodes"],
                    max_nodes=60,
                    title=f"Path Trace: {src_input} → {dst_input}"
                )
                st.plotly_chart(fig_path, width="stretch")
                
                # Hops detail table
                st.markdown("**Hop-by-Hop Monetary Transfers:**")
                st.dataframe(pd.DataFrame(path_res["paths"][0]["hops"]), width="stretch")
            elif path_res["status"] == "not_connected":
                st.info("No directed path connects these two accounts within the maximum depth cutoff.")
            else:
                st.error(path_res.get("message", "Error finding path."))
                
    with net_tab_topo:
        st.subheader("Suspicious Anti-Money Laundering (AML) Topologies")
        patterns = detect_suspicious_network_patterns(G, min_fan_degree=3)
        
        top1, top2 = st.columns(2)
        with top1:
            st.markdown(f"**Fan-in Structuring / Pooling Nodes ({len(patterns['fan_in_patterns'])} detected):**")
            st.caption("Multiple distinct senders channeling funds to a single central accumulator account.")
            if patterns["fan_in_patterns"]:
                st.dataframe(pd.DataFrame(patterns["fan_in_patterns"]).head(10), width="stretch")
            else:
                st.info("No high-degree fan-in structures detected.")
                
        with top2:
            st.markdown(f"**Fan-out Dispersal / Layering Nodes ({len(patterns['fan_out_patterns'])} detected):**")
            st.caption("A single source distributing capital outward to multiple distinct accounts.")
            if patterns["fan_out_patterns"]:
                st.dataframe(pd.DataFrame(patterns["fan_out_patterns"]).head(10), width="stretch")
            else:
                st.info("No high-degree fan-out structures detected.")


# ==============================================================================
# PAGE 6: ANOMALY DETECTION
# ==============================================================================
elif selected_page == "6. Anomaly Detection":
    st.markdown('<div class="hero-badge">Module 4 & 7 • Forensic Anomaly Isolation</div>', unsafe_allow_html=True)
    st.title("Suspicious Account Candidates & Explainability")
    st.markdown("Detailed forensic audit explaining the exact behavioral drivers behind each flagged account.")
    
    # Generate full report
    full_report_df = generate_forensic_investigation_report(features_df, ml_results, top_n=200)
    
    # Filter controls
    f_col1, f_col2 = st.columns([1, 2])
    score_thresh = f_col1.slider("Minimum Anomaly Score Threshold", 0.0, 1.0, 0.5, 0.05)
    selected_tier = f_col2.multiselect("Filter by Risk Tier", ["CRITICAL", "HIGH", "MEDIUM"], default=["CRITICAL", "HIGH"])
    
    filtered_report = full_report_df[
        (full_report_df["Max Anomaly Score"] >= score_thresh) &
        (full_report_df["Risk Tier"].isin(selected_tier))
    ]
    
    st.metric("Suspects Meeting Criteria", f"{len(filtered_report)} accounts")
    st.dataframe(filtered_report, width="stretch")
    
    # Account Explainability Deep Dive
    st.markdown("---")
    st.subheader("🔬 Account Explainability Deep-Dive")
    if not filtered_report.empty:
        deep_acc = st.selectbox("Select Account for Detailed Explanation", filtered_report["Account ID"].unique())
        acc_row = filtered_report[filtered_report["Account ID"] == deep_acc].iloc[0]
        
        d1, d2, d3 = st.columns(3)
        d1.metric("Risk Level", acc_row["Risk Tier"])
        d2.metric("Max Anomaly Score", acc_row["Max Anomaly Score"])
        d3.metric("Flagging Models", acc_row["Algorithms Flagging"])
        
        st.markdown(f"**Forensic Explanation:**\n> {acc_row['Primary Forensic Explanations']}")
        
        # Behavioral Bar Chart for this account vs normal population
        acc_feats = features_df.loc[deep_acc]
        mean_feats = features_df.mean()
        
        compare_df = pd.DataFrame({
            "Metric": ["Total Sent", "Total Received", "Avg Txn Size", "Turnover"],
            "This Account": [acc_feats["total_sent"], acc_feats["total_received"], acc_feats["overall_avg_amount"], acc_feats["total_turnover"]],
            "Population Average": [mean_feats["total_sent"], mean_feats["total_received"], mean_feats["overall_avg_amount"], mean_feats["total_turnover"]]
        })
        fig_comp = px.bar(
            compare_df, x="Metric", y=["This Account", "Population Average"],
            barmode="group", template="plotly_dark",
            color_discrete_sequence=["#FF3366", "#00F0FF"]
        )
        fig_comp.update_layout(paper_bgcolor="#1E232F", plot_bgcolor="#131722", height=350)
        st.plotly_chart(fig_comp, width="stretch")


# ==============================================================================
# PAGE 7: PCA VS T-SNE
# ==============================================================================
elif selected_page == "7. PCA vs t-SNE":
    st.markdown('<div class="hero-badge">Module 5 • Manifold Learning & Dimension Reduction</div>', unsafe_allow_html=True)
    st.title("PCA vs t-SNE Dimensionality Reduction")
    st.markdown(
        "Compare linear Principal Component Analysis (global variance maximization) against "
        "non-linear t-SNE (local neighborhood manifold clustering)."
    )
    
    col_t1, col_t2 = st.columns([1, 2])
    perp_slider = col_t1.slider("t-SNE Perplexity", min_value=10.0, max_value=50.0, value=30.0, step=5.0)
    col_t2.info(
        "**Viva Theory:**\n"
        "- **PCA** is linear; preserves overall variance and global distances. Components are orthogonal.\n"
        "- **t-SNE** is non-linear; converts Euclidean distances to joint probabilities and minimizes KL-divergence. "
        "It excels at clustering dense local neighborhoods and separating isolated anomalies."
    )
    
    with st.spinner("Computing PCA and t-SNE 2D projections..."):
        pca_data = compute_pca(scaled_df.values, n_components=2)
        tsne_data = compute_tsne(scaled_df.values, n_components=2, perplexity=perp_slider, max_samples=1500)
        
    p1, p2 = st.columns(2)
    p1.metric("PCA Explained Variance (2D)", f"{pca_data['total_variance_explained']*100:.2f}%")
    p2.metric("t-SNE Subsampled Points", f"{len(tsne_data['sample_indices'])} accounts")
    
    st.markdown("###")
    
    fig_reduction = plot_pca_vs_tsne_side_by_side(
        pca_data,
        tsne_data,
        account_ids=list(scaled_df.index),
        labels=ml_results["K-Means"]["labels"],
        anomaly_flags=ml_results["Isolation Forest"]["flags"]
    )
    st.plotly_chart(fig_reduction, width="stretch")


# ==============================================================================
# PAGE 8: INVESTIGATION REPORTS
# ==============================================================================
elif selected_page == "8. Reports":
    st.markdown('<div class="hero-badge">Module 7 • Audit Exports & Case Summaries</div>', unsafe_allow_html=True)
    st.title("Financial Crime Investigation Reports")
    st.markdown("Generate, preview, and export compliance audit reports and suspect accounts dossier.")
    
    report_df = generate_forensic_investigation_report(features_df, ml_results, top_n=250)
    csv_data = export_report_to_csv(report_df)
    
    # Executive Summary Metrics
    c_crit = sum(report_df["Risk Tier"] == "CRITICAL")
    c_high = sum(report_df["Risk Tier"] == "HIGH")
    total_exposure = report_df["Total Sent ($)"].sum()
    
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Total Suspects Domiciled", len(report_df))
    r2.metric("Critical Risk Accounts", c_crit)
    r3.metric("High Risk Accounts", c_high)
    r4.metric("Total Capital Exposure", f"${total_exposure:,.2f}")
    
    st.markdown("###")
    st.subheader("📥 Export Official Case Reports")
    st.download_button(
        label="Download Full Investigation Report (CSV)",
        data=csv_data,
        file_name="FraudLink_Investigation_Audit_Report.csv",
        mime="text/csv",
        type="primary"
    )
    
    st.markdown("###")
    st.subheader("Forensic Dossier Table")
    st.dataframe(report_df, width="stretch")
    
    st.markdown("---")
    with st.expander("📚 College Viva Preparation Guide: INT396", expanded=False):
        st.markdown("""
        ### Key Viva Questions & Answers:
        1. **What is unsupervised learning and why is it used in fraud detection?**
           *Supervised models require labeled fraud data which is scarce, biased, and quickly obsolete as fraudsters evolve new techniques. Unsupervised learning identifies anomalies and novel patterns directly from intrinsic data geometry without labels.*
        2. **Why do we use both Isolation Forest and DBSCAN?**
           *Isolation Forest excels in high-dimensional tabular spaces by measuring tree isolation depth. DBSCAN excels in dense spatial clusters and isolates noise points of arbitrary topological shapes.*
        3. **Why do we need NetworkX in addition to Machine Learning?**
           *Machine learning identifies suspicious accounts individually based on their behavioral features. NetworkX traces the multi-hop relationships between accounts, proving whether flagged accounts are part of an organized laundering syndicate or mule ring.*
        """)
