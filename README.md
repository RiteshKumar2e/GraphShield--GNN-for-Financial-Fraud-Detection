# GraphShield — Explainable Graph Neural Network for Financial Fraud Detection

<p align="center">
  <img src="results/figures/transaction_graph.png" alt="Bitcoin Transaction Graph" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.12-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyG-2.6.1-3c9cd7?logo=pyg&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Dataset-Elliptic%20Bitcoin-orange" />
</p>

> **GraphShield** models financial transactions as a graph and applies Graph Neural Networks — GCN, GraphSAGE, and GAT — to detect illicit Bitcoin transactions. GNNExplainer surfaces the suspicious subgraphs that drive each prediction, making the system auditable as well as accurate.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Graph Structure](#graph-structure)
- [Architecture](#architecture)
- [Results](#results)
- [Explainability](#explainability)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Notebook Guide](#notebook-guide)
- [Key Findings](#key-findings)
- [Citation](#citation)

---

## Overview

Traditional fraud detection treats every transaction independently using tabular features. Real-world fraud is **network-based** — fraudsters share devices, IP addresses, and merchants, and route money through rings of accounts.

GraphShield captures these patterns by:

1. Representing transactions as **nodes** and money flows as **edges** in a directed graph
2. Training GNN models that aggregate signals from neighbouring transactions
3. Comparing against 5 ML baselines to isolate the contribution of graph structure
4. Explaining predictions with **GNNExplainer** — surfacing the exact subgraph and features driving each fraud flag

---

## Dataset

**Elliptic Bitcoin Dataset** — the largest publicly labelled Bitcoin transaction graph.

| Property | Value |
|---|---|
| Total nodes | 203,769 |
| Labeled nodes used | **46,564** |
| Illicit (fraud) | 4,545 (9.77%) |
| Licit (clean) | 41,941 (90.23%) |
| Edges | **36,624** |
| Node features | 166 (94 local + 72 aggregated neighbourhood) |
| Time steps | 49 |

Download from Kaggle: [ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)  
Place the three CSV files in `data/raw/`.

<p align="center">
  <img src="results/figures/graph_overview.png" alt="Dataset Overview" width="100%"/>
</p>

---

## Graph Structure

Each **node** is a Bitcoin transaction. A directed **edge** from node *u* → *v* means transaction *u* is an input to transaction *v* (i.e., BTC flows from *u* to *v*).

```
Transaction A ──▶ Transaction C ──▶ Transaction E
Transaction B ──▶ Transaction C
Transaction D ──▶ Transaction E
```

Node features include local statistics (transaction amount, number of inputs/outputs, fees) and 72 aggregated one-hop neighbourhood statistics. Labels are **illicit** (fraud) or **licit** (clean); ~79% of the full graph is unlabelled.

---

## Architecture

### GNN Models

```
Input Node Features  [N × 166]
        │
   ┌────▼────┐
   │  Conv 1  │  GCNConv / SAGEConv / GATConv (heads=4)
   │  ReLU    │
   │  Dropout │  p = 0.4
   └────┬────┘
        │
   ┌────▼────┐
   │  Conv 2  │  Output dim = 2
   └────┬────┘
        │
   Softmax ──▶ {Licit, Illicit}
```

| Model | Aggregation | Key Property |
|---|---|---|
| **GCN** | Spectral normalised sum | Fast, strong baseline |
| **GraphSAGE** | Sampled mean | Inductive, scalable |
| **GAT** | Learned attention (4 heads) | Identifies important neighbours |
| **DeepGCN** | 3-layer + BatchNorm | Ablation depth study |

### Training

- **Optimiser:** Adam — lr 0.001, weight_decay 5×10⁻⁴
- **Loss:** Weighted cross-entropy (inverse-frequency class weights)
- **Epochs:** 200
- **Split:** Stratified 64 / 16 / 20 % (train / val / test)

---

## Results

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.878 | 0.441 | 0.929 | 0.598 | 0.965 | 0.755 |
| MLP | 0.981 | 0.938 | 0.864 | 0.899 | 0.986 | 0.941 |
| Random Forest | 0.988 | 0.996 | 0.876 | 0.932 | 0.997 | 0.982 |
| LightGBM | 0.993 | 0.987 | 0.937 | 0.962 | **0.998** | **0.990** |
| XGBoost | 0.993 | 0.993 | 0.933 | **0.962** | 0.997 | 0.987 |
| GCN | 0.907 | 0.513 | 0.896 | 0.653 | 0.965 | 0.810 |
| GraphSAGE | 0.936 | 0.618 | 0.899 | 0.732 | 0.977 | 0.900 |
| **GAT** | 0.874 | 0.434 | **0.952** | 0.596 | 0.976 | 0.868 |

> **Key insight:** GAT achieves the highest recall (0.952) of all models including all ML baselines — it misses the fewest fraud cases. Tree models win on F1 because Elliptic node features already include pre-aggregated graph statistics; GNNs derive this from raw edges and show their strongest advantage in settings without such feature engineering.

### Combined ROC & PR Curves — GNN Models

<p align="center">
  <img src="results/figures/gnn_roc_curves.png" width="48%" />
  <img src="results/figures/gnn_pr_curves.png" width="48%" />
</p>

### Training Curves

<p align="center">
  <img src="results/figures/gcn_training_history.png" width="32%" />
  <img src="results/figures/graphsage_training_history.png" width="32%" />
  <img src="results/figures/gat_training_history.png" width="32%" />
</p>

### Confusion Matrices

<p align="center">
  <img src="results/confusion_matrices/gcn_cm.png" width="32%" />
  <img src="results/confusion_matrices/graphsage_cm.png" width="32%" />
  <img src="results/confusion_matrices/gat_cm.png" width="32%" />
</p>

---

## Ablation Study

### Effect of Class Weights

| Configuration | F1 | Recall | PR-AUC | ROC-AUC |
|:---|---:|---:|---:|---:|
| GCN + Class Weights | 0.632 | **0.887** | **0.796** | **0.961** |
| GCN − Class Weights | 0.729 | 0.618 | 0.794 | 0.941 |

Removing class weights raises F1 but recall drops from 0.887 → 0.618 — the model misses 28% more fraud cases.

### Effect of Model Depth

| Configuration | F1 | Recall | PR-AUC |
|:---|---:|---:|---:|
| GCN 2-Layer | 0.608 | 0.891 | 0.779 |
| GCN 3-Layer | 0.595 | **0.904** | 0.781 |

<p align="center">
  <img src="results/figures/ablation_study.png" width="70%" />
</p>

### Temporal Split

Training on time steps 1–4, testing on 5–49 (simulates real deployment):

| Split | F1 | PR-AUC |
|:---|---:|---:|
| Random | 0.632 | 0.796 |
| Temporal | 0.298 | 0.519 |

Fraud patterns shift over time — motivation for temporal GNN extensions.

---

## Explainability

GNNExplainer is applied to the highest-confidence fraud prediction (node 7929, fraud probability **0.9994**).

<p align="center">
  <img src="results/figures/explanation_subgraph.png" width="55%" />
  <img src="results/figures/feature_importance.png" width="42%" />
</p>

**Finding:** Node 7929 is flagged primarily because of its dense connectivity to a cluster of known illicit nodes. The top predictive features are neighbourhood-aggregated statistics (features 93–165) — not individual transaction amounts — consistent with fraud ring behaviour.

---

## Project Structure

```
GraphShield/
├── data/
│   ├── raw/                         # Elliptic CSVs (download from Kaggle)
│   └── processed/                   # Generated .pt tensors (run notebook 02)
│
├── notebooks/
│   ├── 01_data_loading_and_eda.ipynb
│   ├── 02_graph_construction.ipynb
│   ├── 03_ml_baseline_models.ipynb
│   ├── 04_gcn_model.ipynb
│   ├── 05_graphsage_model.ipynb
│   ├── 06_gat_model.ipynb
│   ├── 07_model_comparison.ipynb
│   ├── 08_explainability_gnnexplainer.ipynb
│   └── 09_research_results_and_ablation.ipynb
│
├── src/
│   ├── data_preprocessing.py        # Load, clean, normalize Elliptic data
│   ├── graph_builder.py             # Build PyG Data object, masks, save/load
│   ├── models.py                    # GCN, GraphSAGE, GAT, DeepGCN
│   ├── train.py                     # Training loop with class weights
│   ├── evaluate.py                  # Metrics, ROC/PR curves, confusion matrix
│   ├── explain.py                   # GNNExplainer wrapper, subgraph viz
│   ├── utils.py                     # Seed, device, save/load model
│   └── visualize_graph.py           # Transaction graph & dataset overview figs
│
├── models/                          # Saved model weights (.pt)
├── results/
│   ├── figures/                     # All plots and visualisations
│   ├── confusion_matrices/
│   ├── comparison_table.csv
│   ├── baseline_metrics.csv
│   └── ablation_results.csv
│
├── paper/
│   ├── abstract.md
│   ├── introduction.md
│   ├── literature_review.md
│   ├── methodology.md
│   ├── results.md
│   └── conclusion.md
│
├── config.yaml                      # All hyperparameters in one place
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/RiteshKumar2e/GraphShield-Explainable-Graph-Neural-Network-for-Financial-Fraud-Detection.git
cd GraphShield-Explainable-Graph-Neural-Network-for-Financial-Fraud-Detection
pip install -r requirements.txt
```

> **PyTorch Geometric** requires a separate install matched to your CUDA version.  
> See: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

### 2. Download dataset

Download from [Kaggle — Elliptic Bitcoin Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) and place all three files in `data/raw/`:

```
data/raw/
├── elliptic_txs_features.csv
├── elliptic_txs_classes.csv
└── elliptic_txs_edgelist.csv
```

### 3. Run notebooks in order

```
01 → EDA and data understanding
02 → Graph construction (saves .pt files to data/processed/)
03 → ML baseline models
04 → GCN training
05 → GraphSAGE training
06 → GAT training
07 → Full model comparison + figures
08 → GNNExplainer
09 → Ablation study + paper tables
```

### 4. Regenerate graph visualisations

```bash
python src/visualize_graph.py
```

---

## Notebook Guide

| # | Notebook | What it does |
|---|---|---|
| 01 | Data Loading & EDA | Class distribution, feature stats, time steps, graph statistics |
| 02 | Graph Construction | Builds PyG `Data` object, stratified masks, saves to disk |
| 03 | ML Baselines | LR, RF, XGBoost, LightGBM, MLP — fraud metrics & comparison chart |
| 04 | GCN | 2-layer GCN with class weights — trains & evaluates |
| 05 | GraphSAGE | Inductive sampling-based GNN |
| 06 | GAT | 4-head graph attention network |
| 07 | Model Comparison | Combined ROC/PR curves, full comparison table |
| 08 | GNNExplainer | Explanation subgraph + top feature importances |
| 09 | Ablation Study | Class weights, model depth, temporal split |

---

## Key Findings

| Finding | Detail |
|---|---|
| **GNNs maximise recall** | GAT recall **0.952** — highest of all 8 models |
| **Class weighting is essential** | Without it, recall drops from 0.887 → 0.618 |
| **GraphSAGE best GNN overall** | F1 0.732, PR-AUC 0.900, ROC-AUC 0.977 |
| **Temporal gap is significant** | PR-AUC drops from 0.796 → 0.519 under temporal split |
| **Fraud is structural** | GNNExplainer shows neighbourhood topology dominates over individual tx features |

---

## Citation

If you use this project, please cite the Elliptic dataset:

```bibtex
@inproceedings{weber2019antimoney,
  title     = {Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  author    = {Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E},
  booktitle = {KDD Workshop on Anomaly Detection in Finance},
  year      = {2019}
}
```

And the GNN architectures used:

```bibtex
@inproceedings{kipf2017semi,
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  author    = {Kipf, Thomas N and Welling, Max},
  booktitle = {ICLR},
  year      = {2017}
}

@inproceedings{hamilton2017inductive,
  title     = {Inductive Representation Learning on Large Graphs},
  author    = {Hamilton, William L and Ying, Rex and Leskovec, Jure},
  booktitle = {NeurIPS},
  year      = {2017}
}

@inproceedings{velickovic2018graph,
  title     = {Graph Attention Networks},
  author    = {Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle = {ICLR},
  year      = {2018}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">Built with PyTorch · PyTorch Geometric · NetworkX · scikit-learn</p>
