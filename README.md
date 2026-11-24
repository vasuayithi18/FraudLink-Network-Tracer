# FraudLink + Network Tracer
**Subject: INT396 – Unsupervised Learning**

A unified financial crime investigation system combining **Unsupervised Machine Learning**, **Entity Behavioral Profiling**, **Graph Intelligence (NetworkX)**, and an **Interactive Forensic Dashboard (Streamlit & Plotly)**.

---

## 🌟 Key Features

1. **Overview Command Center (Dashboard)**
   - Real-time KPIs: Total transactions, unique accounts, total volume ($), and anomaly candidate counts.
   - Interactive breakdown of transaction types (PAYMENT, TRANSFER, CASH_OUT, CASH_IN, DEBIT).
   - Anomaly score frequency distribution and priority suspect candidate cards.

2. **Dataset Explorer**
   - Out-of-the-box support for the PaySim dataset schema as well as custom CSV uploads.
   - Intelligent sampling for large files (supports 500MB+ logs without RAM exhaustion).
   - Missing value and duplicate audits with automated cleaning routines.

3. **Entity Behavioral Preprocessing**
   - Transforms transaction logs into **Account Profiles** over time (18 behavioral dimensions).
   - Flow metrics ($Net Flow = Received - Sent$), turnover ratios, counterparty degree.
   - Anti-Money Laundering (AML) signatures: Balance draining count and high-risk transfer counts.
   - `StandardScaler` normalization ($\mu = 0, \sigma = 1$) to prevent monetary scale bias in distance-based models.
   - **Zero Target Leakage**: Fraud labels (`isFraud`, `isFlaggedFraud`) are strictly segregated from training.

4. **Unsupervised Machine Learning Engine**
   - **K-Means Clustering**: With automated Elbow curve inertia and Silhouette score analysis across $K$.
   - **DBSCAN**: Density-based clustering isolating core, border, and noise anomaly points ($\varepsilon$, $MinPts$).
   - **Hierarchical Clustering**: Agglomerative clustering with Ward linkage and Dendrogram visualization.
   - **Isolation Forest**: Recursive random partitioning measuring anomaly path depths.
   - **Local Outlier Factor (LOF)**: Measuring local density deviation relative to $k$-nearest neighbors.
   - **Master Benchmarking Matrix**: Comparing Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.

5. **Network Tracer & Graph Intelligence**
   - Directed transaction graph (`NetworkX DiGraph`) with monetary edge weights and timestamp attributes.
   - Account search by ID with forensic profiles and 1-hop Ego-Network visualization.
   - Multi-hop transaction path tracing (discovering money laundering chains and intermediary mules).
   - AML topology mining: Fan-in (structuring/pooling), Fan-out (dispersal/layering), and circular money loops.
   - Interactive Plotly network visualizations with color-coded risk highlights.

6. **Explainable Anomaly Detection**
   - Transparent, plain-language forensic reasoning explaining *why* an account was flagged.
   - Rule-based triggers (balance wiped to \$0, extreme unidirectional outflow, rapid pass-through mule behavior).
   - Risk tiering: `CRITICAL`, `HIGH`, `MEDIUM`.

7. **Dimensionality Reduction: PCA vs t-SNE**
   - Linear global variance projection (**PCA**) vs. non-linear local manifold clustering (**t-SNE**).
   - Side-by-side interactive 2D scatter plots color-coded by cluster and anomaly status.

8. **Compliance & Investigation Reports**
   - Comprehensive forensic dossier table with capital exposure calculations.
   - One-click CSV export for official audit reports.
   - Embedded INT396 viva preparation guide.

---

## 🚀 How to Run the Application

Execute the following command in PowerShell from the project root:

```powershell
& .\.venv\Scripts\python.exe -m streamlit run app.py
```

Open your browser at: **`http://localhost:8501`**

---

## 📁 Project Structure

```text
FraudLink-Network-Tracer/
│
├── app.py                             # Streamlit Main Web Application Entrypoint
├── requirements.txt                   # Pinned Dependencies
├── README.md                          # Project Documentation & Viva Guide
│
├── data/
│   ├── generate_sample_data.py        # Realistic PaySim generator script
│   ├── paysim_sample.csv              # Ready-to-use sample dataset (4,791 rows)
│   └── transactions.csv               # Full real PaySim dataset (476.7 MB)
│
├── preprocessing/
│   ├── loader.py                      # Fast chunked loader, cleaning & schema audit
│   ├── feature_engineering.py         # 18 account behavioral features
│   └── scaler.py                      # Scikit-learn StandardScaler normalization
│
├── models/
│   ├── clustering.py                  # K-Means (Elbow), DBSCAN, Hierarchical
│   ├── anomaly.py                     # Isolation Forest & Local Outlier Factor (LOF)
│   └── evaluation.py                  # Silhouette, Davies-Bouldin, Calinski-Harabasz
│
├── network/
│   ├── graph_builder.py               # NetworkX DiGraph builder & topological metrics
│   ├── tracer.py                      # Multi-hop path tracer & AML topology miner
│   └── network_viz.py                 # Interactive Plotly network & ego-network graphs
│
├── visualization/
│   ├── model_plots.py                 # Elbow, Dendrogram, Anomaly score distributions
│   └── dimensionality_reduction.py    # PCA and t-SNE 2D projections
│
├── reports/
│   └── reporter.py                    # Forensic explainability engine & CSV exporter
│
├── test_phase1_2.py                   # Automated tests for Phase 1 & 2
├── test_phase3.py                     # Automated tests for Phase 3 (ML models)
├── test_phase4.py                     # Automated tests for Phase 4 (Network Tracer)
└── test_app_pages.py                  # Automated tests for all 8 web app pages
```

---

## 🎓 College Viva Preparation (INT396)

1. **Why do we aggregate transactions into account-level features?**
   *Single transactions only show instantaneous values. Financial crime (mule rings, structuring, layering) is an entity-level behavioral phenomenon across multiple transactions.*

2. **Why is `StandardScaler` mandatory for unsupervised clustering?**
   *Distance-based algorithms (K-Means, DBSCAN, Hierarchical, LOF) use Euclidean distance. Without standardization, monetary amounts (\$10 to \$1,000,000) would completely overwhelm transaction frequencies (1 to 10), blinding the model to transaction velocity and topology.*

3. **Why must `isFraud` never be an input feature during model training?**
   *Including ground truth labels in unsupervised models causes data leakage. Real-world financial systems must detect zero-day fraud patterns where labels do not exist yet.*

4. **What is the difference between PCA and t-SNE?**
   *PCA is a linear technique that preserves global variance along orthogonal eigenvectors. t-SNE is a non-linear manifold technique that preserves local probabilistic neighborhoods, making it superior for visualizing distinct clusters and isolated outliers.*
