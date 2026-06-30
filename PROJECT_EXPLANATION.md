# GraphShield — Project Explanation

## Ye Project Kya Hai?

**GraphShield** ek AI-based system hai jo **Bitcoin transactions mein financial fraud detect karta hai** — aur sirf detect hi nahi karta, balki **explain bhi karta hai ki fraud kyun pakda gaya**.

Puri duniya mein financial fraud se har saal **hundreds of billions of dollars** ka nuksan hota hai. Is project ka goal tha ek aisa system banana jo:

1. Fraud transactions ko **accurately identify** kar sake
2. Model ka decision **explainable** ho (black box nahi)
3. Traditional ML se **behtar** kaam kare jab relationships matter karte ho

---

## Kyun Banaya Gaya? (Problem Statement)

### Traditional ML ki Kami

Pehle ke fraud detection systems jaise **Logistic Regression, Random Forest, XGBoost** har transaction ko **alag-alag (independently)** dekhte the.

**Problem:** Real-world mein fraud aisa nahi hota. Fraudsters:
- Ek hi device/IP address se multiple fake accounts chalate hain
- **Fraud rings** banate hain — paise ek account se doosre mein transfer hote rahte hain
- Akela transaction ekdum normal lagta hai, lekin uski **connections** dekhne par suspicious lagta hai

```
Normal transaction dikhta hai:
   [Tx A] = ₹500 transfer  ← looks clean individually

Lekin graph mein dekhne par:
   [Tx A] ──▶ [Tx C (FRAUD)] ──▶ [Tx E (FRAUD)]
   [Tx B (FRAUD)] ──▶ [Tx C]
   ← ab Tx A bhi suspicious lag raha hai!
```

### Graph Neural Network ka Solution

**GNNs** transactions ko ek **graph** ki tarah model karte hain:
- Har **node** = ek Bitcoin transaction
- Har **edge** = BTC ka flow ek transaction se doosre mein

GNN neighboring nodes ka information propagate karta hai — agar koi transaction bahut saare known-fraud nodes se connected hai, toh us transaction ko bhi high fraud score milta hai, **chahe uske apne features normal ho**.

---

## Project ka Architecture

### Dataset — Elliptic Bitcoin Dataset

| Property | Value |
|---|---|
| Total Nodes | 203,769 Bitcoin transactions |
| Labeled Nodes (use kiye) | **46,564** |
| Fraud (Illicit) | 4,545 — **9.77%** |
| Clean (Licit) | 41,941 — **90.23%** |
| Edges | 36,624 |
| Node Features | 166 (94 local + 72 neighbourhood stats) |
| Time Steps | 49 discrete snapshots |

Yeh duniya ka **sabse bada publicly labeled Bitcoin transaction graph** hai.

---

### Kaunse Models Use Kiye?

#### Traditional ML Baselines (comparison ke liye)
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- MLP (Neural Network)

#### Graph Neural Networks (main models)
| Model | Kya karta hai |
|---|---|
| **GCN** (Graph Convolutional Network) | Sabhi neighbors se normalized sum leke features aggregate karta hai |
| **GraphSAGE** | Random sample kiye neighbours ka mean leke aggregate karta hai — bade graphs par bhi kaam karta hai |
| **GAT** (Graph Attention Network) | Attention mechanism se decide karta hai ki **kaun sa neighbour zyada important hai** (4 attention heads) |
| **DeepGCN** | 3-layer GCN — depth ka effect study karne ke liye (ablation) |

---

### Class Imbalance Problem

Dataset mein sirf **9.77% fraud** hai aur 90.23% clean transactions hain. Agar model seedha train karo, toh woh har transaction ko "clean" bolna seekh lega (easy shortcut).

**Solution:** Inverse-frequency class weights — fraud class ko zyada weight diya gaya taaki model usse ignore na kare.

```
w_fraud = 1 / count(fraud)   ← zyada weight
w_licit = 1 / count(licit)   ← kam weight
```

---

## Results — Kya Mila?

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.878 | 0.441 | 0.929 | 0.598 | 0.965 | 0.755 |
| MLP | 0.981 | 0.938 | 0.864 | 0.899 | 0.986 | 0.941 |
| Random Forest | 0.988 | 0.996 | 0.876 | 0.932 | 0.997 | 0.982 |
| LightGBM | 0.993 | 0.987 | 0.937 | **0.962** | **0.998** | **0.990** |
| XGBoost | 0.993 | 0.993 | 0.933 | **0.962** | 0.997 | 0.987 |
| GCN | 0.907 | 0.513 | 0.896 | 0.653 | 0.965 | 0.810 |
| GraphSAGE | 0.936 | 0.618 | 0.899 | 0.732 | 0.977 | 0.900 |
| **GAT** | 0.874 | 0.434 | **0.952** | 0.596 | 0.976 | 0.868 |

> **Recall sabse important metric hai fraud detection mein** — ek missed fraud ek false alarm se kahin zyada costly hai.
> GAT ne sabse highest recall (0.952) achieve kiya — sabse kam fraud cases miss kiye.

---

## Explainability — GNNExplainer

Sirf fraud detect karna kaafi nahi — **kyun fraud hai yeh bhi batana zaroori hai**.

**GNNExplainer** har prediction ke liye ek "explanation subgraph" generate karta hai — woh exact transactions jo model ne fraud decide karne mein use kiye.

**Example:** Node 7929 (fraud probability = 0.9994)
- Model ne dekha ki yeh node ek dense cluster of illicit nodes se connected hai
- Top predictive features **neighbourhood-aggregated statistics** the, individual transaction amount nahi
- Matlab: fraud ka signal individual transaction mein nahi tha — **uski connections mein tha**

Isse fraud analysts ko pata chalta hai **ki kyun** transaction flag hua, na sirf **ki** hua.

---

## Ablation Study — Kya Seekha?

### 1. Class Weights Zaroori Hain

| Config | Recall | F1 |
|:---|---:|---:|
| GCN + Class Weights | **0.887** | 0.632 |
| GCN - Class Weights | 0.618 | 0.729 |

Class weights hatane par recall 28% gir gayi — model fraud bhoolne lag gaya.

### 2. Temporal Split — Real World Ka Sach

| Split Type | PR-AUC |
|:---|---:|
| Random Split | 0.796 |
| Temporal Split (train: steps 1-34, test: 35-49) | **0.519** |

Fraud patterns time ke saath **change** hote hain. Random split unrealistically high performance dikhata hai. Real deployment mein temporal evaluation zyada sahi hai.

---

## Technology Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.12 |
| Graph ML | PyTorch Geometric (PyG) 2.6.1 |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Explainability | GNNExplainer (built-in PyG) |
| Visualization | matplotlib, seaborn, networkx |
| Data | Elliptic Bitcoin Dataset (Kaggle) |

---

## Project Structure Overview

```
GraphShield/
├── data/
│   ├── raw/          ← Elliptic CSVs (Kaggle se download)
│   └── processed/    ← .pt tensors (graph data)
│
├── notebooks/        ← 9 Jupyter notebooks (step-by-step workflow)
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
├── src/              ← Python modules
│   ├── data_preprocessing.py   ← Data load + normalize
│   ├── graph_builder.py        ← PyG graph banana
│   ├── models.py               ← GCN, GraphSAGE, GAT, DeepGCN
│   ├── train.py                ← Training loop
│   ├── evaluate.py             ← Metrics + curves
│   └── explain.py              ← GNNExplainer wrapper
│
├── paper/            ← Research paper sections (markdown)
├── results/          ← Figures, confusion matrices, CSVs
└── config.yaml       ← Hyperparameters
```

---

## Key Findings Summary

| Finding | Result |
|---|---|
| Sabse zyada Recall (sab models mein) | **GAT — 0.952** |
| Sabse zyada F1 (sab models mein) | **XGBoost / LightGBM — 0.962** |
| Best GNN overall | **GraphSAGE — F1 0.732, PR-AUC 0.900** |
| Class weights critical | Recall 0.887 → 0.618 bina class weights ke |
| Temporal gap serious hai | PR-AUC 0.796 → 0.519 temporal split mein |
| Fraud structural hai | Top features neighbourhood stats hain, individual transaction amount nahi |

---

## Future Work

1. **Temporal GNN (TGN/TGAT)** — Time ke saath evolve hote fraud patterns handle karna
2. **Heterogeneous Graph** — Users, devices, IPs, merchants — sab alag node types as R-GCN or HGT
3. **Self-Supervised Learning** — Kam labels ke saath bhi kaam karna
4. **Federated GNN** — Multiple banks milke train karein bina data share kiye
5. **Real-Time Detection** — Streaming transactions mein sub-second fraud scoring
6. **Adversarial Robustness** — Fraudsters jo deliberately graph structure manipulate karein unhe handle karna

---

## Ek Line Summary

> **GraphShield isliye banaya gaya kyunki real-world financial fraud sirf numbers mein nahi, balki connections mein hota hai — aur GNNs woh connections samajhte hain jo traditional ML nahi samajh sakta.**
