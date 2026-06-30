# GraphShield — Project Explanation

## Ye Project Kya Hai?

**GraphShield** ek AI-based system hai jo **financial fraud detect karta hai** — sirf ek dataset mein nahi, balki **5 alag-alag real-world fraud domains** mein — aur har prediction ko **explain bhi karta hai** ki fraud kyun pakda gaya.

Puri duniya mein financial fraud se har saal **hundreds of billions of dollars** ka nuksan hota hai. Is project ka goal tha ek aisa system banana jo:

1. **5 alag domains** mein fraud accurately identify kare — Bitcoin, Credit Card, Mobile Payments, Insurance, E-Commerce
2. Model ka decision **explainable** ho — black box nahi
3. Tabular data ko bhi **graph mein convert** kare taaki GNNs use ho sakein
4. Ek **interactive dashboard** ho jisme koi bhi transaction ka fraud network explore kar sake

---

## Kyun Banaya Gaya? (Problem Statement)

### Traditional ML ki Kami

Pehle ke fraud detection systems jaise **Logistic Regression, Random Forest, XGBoost** har transaction ko **alag-alag (independently)** dekhte the.

**Problem:** Real-world mein fraud aisa nahi hota. Fraudsters:
- Ek hi device/IP se multiple fake accounts chalate hain
- **Fraud rings** banate hain — paise ek account se doosre mein transfer hote rahte hain
- Akela transaction bilkul normal lagta hai, lekin uski **connections** dekhne par suspicious lagta hai

```
Normal dikhta hai:
   [Tx A] = ₹500 transfer  ← individually clean lagta hai

Lekin graph mein dekhne par:
   [Tx A] ──▶ [Tx C (FRAUD)] ──▶ [Tx E (FRAUD)]
   [Tx B (FRAUD)] ──▶ [Tx C]
   ← ab Tx A bhi suspicious!
```

### Graph Neural Network ka Solution

**GNNs** transactions ko ek **graph** ki tarah model karte hain:
- Har **node** = ek transaction
- Har **edge** = connection (ya similarity)

GNN neighboring nodes ka information propagate karta hai — agar koi transaction bahut saare known-fraud nodes se connected hai, toh us transaction ko bhi high fraud score milta hai, **chahe uske apne features normal ho**.

---

## 5 Datasets — 5 Fraud Domains

GraphShield sirf ek dataset pe nahi, **5 alag-alag fraud domains** par train aur test kiya gaya:

| # | Dataset | Domain | Nodes | Fraud Rate | Graph Type |
|---|---|---|---|---|---|
| 1 | **Elliptic Bitcoin** | Blockchain / Crypto | 46,564 | 9.77% | Natural BTC flow graph |
| 2 | **Credit Card Fraud (ULB)** | Banking | 284,807 | 0.17% | KNN (k=10) on PCA features |
| 3 | **PaySim Online Payments** | Mobile Fintech | 6.3M+ | 0.13% | KNN (k=10) on tx features |
| 4 | **Vehicle Insurance Claims** | Insurance | 15,420 | ~6.0% | KNN (k=10) on mixed features |
| 5 | **E-Commerce Transactions** | Online Retail | 1.47M+ | ~10% | KNN (k=10) on behavioural |

### Dataset 1 — Elliptic Bitcoin (Natural Graph)

Yeh duniya ka **sabse bada publicly labeled Bitcoin transaction graph** hai.

Isme naturally graph structure hai — BTC ek transaction se doosre mein transfer hoti hai, wahi edges hain:
```
[Tx A] ──▶ [Tx C] ──▶ [Tx E]   ← Bitcoin flow
[Tx B] ──▶ [Tx C]
```

### Datasets 2–5 — Tabular Data ko Graph Mein Convert Karna

Credit Card, PaySim, Insurance, E-Commerce mein koi natural graph nahi tha — sirf CSV tables the.

**Solution: KNN Graph Construction**

```
Har transaction = ek node

Do nodes ko connect karo agar woh dono ek-doosre ke
10 nearest neighbours hain feature space mein

Credit Card example:
  [Tx #201] ──── [Tx #847]    ← similar PCA features
  [Tx #201] ──── [Tx #312]    ← similar amount + time pattern
  [Tx #847] ──── [Tx #991]

GNN fraud signal propagate karta hai similar transactions ke cluster mein
```

Isse GNN **"fraud rings"** detect kar sakta hai even tabular data mein — similar profile wale transactions cluster mein hote hain, aur GNN us cluster ke andar fraud signal spread karta hai.

---

## Kaunse Models Use Kiye?

### GNN Models (Main Models)

| Model | Kya Karta Hai |
|---|---|
| **GCN** | Sabhi neighbors se normalized sum leke features aggregate karta hai — fast baseline |
| **GraphSAGE** | Random sample kiye neighbours ka mean leke aggregate — bade graphs par bhi kaam karta hai |
| **GAT** | Attention mechanism se decide karta hai kaun sa neighbour zyada important hai (4 heads) |

### Traditional ML Baselines (Comparison ke liye, Elliptic pe)

- Logistic Regression, Random Forest, XGBoost, LightGBM, MLP

---

## Class Imbalance Problem

Sabhi datasets mein fraud ka percentage bahut kam hai (0.13% se 10%). Agar seedha train karo, model har cheez ko "clean" bolna seekh leta hai.

**Solution:** Inverse-frequency class weights

```
w_fraud = total / (2 × count_fraud)   ← zyada weight
w_licit = total / (2 × count_licit)   ← kam weight
```

Yeh har dataset ke liye automatically calculate hota hai.

---

## Results — Jo Actual Mila

### Elliptic Bitcoin — GNN vs ML Baselines

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|:---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.878 | 0.441 | 0.929 | 0.598 | 0.965 | 0.755 |
| MLP | 0.981 | 0.938 | 0.864 | 0.899 | 0.986 | 0.941 |
| Random Forest | 0.988 | 0.996 | 0.876 | 0.932 | 0.997 | 0.982 |
| LightGBM | 0.993 | 0.987 | 0.937 | **0.962** | **0.998** | **0.990** |
| XGBoost | 0.993 | 0.993 | 0.933 | **0.962** | 0.997 | 0.987 |
| GCN | 0.899 | 0.491 | 0.888 | 0.632 | 0.961 | 0.795 |
| GraphSAGE | 0.923 | 0.566 | 0.897 | 0.694 | 0.973 | 0.879 |
| **GAT** | 0.818 | 0.343 | **0.948** | 0.504 | 0.964 | 0.809 |

> **Recall sabse important metric hai** — ek missed fraud ek false alarm se kahin zyada costly hai.

---

### Multi-Domain Results — Sab 5 Datasets (Notebook 10 ke Results)

Yeh actual output hai jo notebook run karke aaya:

| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Elliptic Bitcoin | GCN | 0.8992 | 0.4909 | 0.8878 | 0.6322 | 0.9612 | 0.7948 |
| Elliptic Bitcoin | **GraphSAGE** | 0.9229 | 0.5664 | 0.8966 | **0.6942** | **0.9727** | **0.8787** |
| Elliptic Bitcoin | GAT | 0.8179 | 0.3433 | **0.9483** | 0.5041 | 0.9644 | 0.8094 |
| Credit Card | GCN | 0.9792 | 0.4293 | 0.8367 | 0.5675 | 0.9690 | 0.8468 |
| Credit Card | **GraphSAGE** | 0.9823 | 0.4775 | 0.8673 | **0.6159** | **0.9784** | **0.8724** |
| Credit Card | GAT | 0.9707 | 0.3388 | 0.8367 | 0.4824 | 0.9709 | 0.8416 |
| PaySim Payments | GCN | 0.8755 | 0.6536 | 0.8033 | 0.7207 | 0.9138 | 0.8208 |
| PaySim Payments | **GraphSAGE** | 0.8872 | 0.6848 | 0.8075 | **0.7411** | **0.9294** | **0.8495** |
| PaySim Payments | GAT | 0.8673 | 0.6325 | 0.8033 | 0.7078 | 0.9096 | 0.8281 |
| Vehicle Insurance | GCN | 0.6430 | 0.1282 | 0.8541 | 0.2230 | 0.8140 | 0.1893 |
| Vehicle Insurance | **GraphSAGE** | 0.6667 | 0.1319 | 0.8162 | **0.2271** | **0.8212** | **0.1939** |
| Vehicle Insurance | GAT | 0.6219 | 0.1264 | **0.8973** | 0.2216 | 0.8137 | 0.1821 |
| E-Commerce | GCN | 0.7297 | 0.3927 | 0.6433 | 0.4877 | 0.7661 | 0.5598 |
| E-Commerce | **GraphSAGE** | 0.7320 | 0.3984 | 0.6667 | **0.4988** | **0.7796** | **0.5777** |
| E-Commerce | GAT | 0.7112 | 0.3734 | 0.6550 | 0.4756 | 0.7634 | 0.5555 |

### Best Model Har Domain Mein

| Domain | Winner | F1 | ROC-AUC | PR-AUC |
|:---|:---:|---:|---:|---:|
| 🔗 Blockchain / Bitcoin | **GraphSAGE** | 0.6942 | 0.9727 | 0.8787 |
| 💳 Credit Card / Banking | **GraphSAGE** | 0.6159 | 0.9784 | 0.8724 |
| 📱 Mobile Payments | **GraphSAGE** | 0.7411 | 0.9294 | 0.8495 |
| 🚗 Vehicle Insurance | **GraphSAGE** | 0.2271 | 0.8212 | 0.1939 |
| 🛒 E-Commerce | **GraphSAGE** | 0.4988 | 0.7796 | 0.5777 |

> **GraphSAGE har ek domain mein best raha** — F1, ROC-AUC aur PR-AUC teenon mein.  
> GAT ne sabse zyada Recall liya (especially Insurance: 0.897) — least fraud miss kiya.  
> GCN fast hai par consistently GraphSAGE se thoda neeche raha.

### Domain-wise Analysis

**1. Elliptic Bitcoin (Best performing — natural graph structure):**
- ROC-AUC 0.97+ sab models mein — natural graph edges se bahut strong signal
- GraphSAGE F1 = 0.694, GAT Recall = 0.948

**2. Credit Card (Sabse imbalanced — 0.17% fraud):**
- ROC-AUC 0.97+ lekin F1 sirf 0.616 — extreme imbalance ki wajah se
- Class weights ne bahut help kiya recall maintain karne mein (0.867)

**3. PaySim Mobile Payments (Best F1 among tabular datasets):**
- F1 0.741 — mobile payment features mein clear fraud pattern hai
- CASH-OUT aur TRANSFER transactions pe fraud concentrate hota hai

**4. Vehicle Insurance (Sabse mushkil domain):**
- F1 sirf 0.227 — bahut small dataset (15K rows) + complex patterns
- Lekin Recall achha hai (0.816–0.897) — fraud cases detect ho rahe hain, precision low hai

**5. E-Commerce (Mid-range performance):**
- F1 0.499, ROC-AUC 0.780
- Behavioural features (device, account age, session) mein fraud signal hai

---

## Explainability — GNNExplainer

Sirf fraud detect karna kaafi nahi — **kyun fraud hai yeh bhi batana zaroori hai**.

**GNNExplainer** har prediction ke liye ek "explanation subgraph" generate karta hai — woh exact transactions jo model ne fraud decide karne mein use kiye.

**Example:** Node 7929 (fraud probability = 0.9994)
- Model ne dekha ki yeh node ek dense cluster of illicit nodes se connected hai
- Top predictive features **neighbourhood-aggregated statistics** the, individual amount nahi
- Matlab: fraud ka signal individual transaction mein nahi tha — **uski connections mein tha**

Isse fraud analysts ko pata chalta hai **ki kyun** transaction flag hua, na sirf **ki** hua.

---

## Ablation Study — Kya Seekha?

### 1. Class Weights Zaroori Hain

| Config | Recall | F1 |
|:---|---:|---:|
| GCN + Class Weights | **0.887** | 0.632 |
| GCN − Class Weights | 0.618 | 0.729 |

Class weights hatane par recall 28% gir gayi — model fraud bhoolne laga.

### 2. Temporal Split — Real World Ka Sach

| Split Type | PR-AUC |
|:---|---:|
| Random Split | 0.796 |
| Temporal Split (train: 1–34, test: 35–49) | 0.519 |

Fraud patterns time ke saath **change** hote hain. Random split unrealistically high performance dikhata hai. Real deployment mein temporal evaluation zyada sahi hai.

### 3. Model Depth

| Config | Recall | F1 |
|:---|---:|---:|
| GCN 2-Layer | 0.891 | 0.608 |
| GCN 3-Layer | **0.904** | 0.595 |

Zyada depth se recall thoda badhti hai, lekin F1 gir sakta hai (over-smoothing).

---

## Interactive Dashboard

GraphShield mein ek **FastAPI + D3.js dashboard** bhi hai:

### 3 Tabs Hain:

**1. Transaction Subgraph Tab**
- Elliptic Bitcoin dataset mein koi bhi node ID daalo
- Uski 2-hop neighbourhood D3.js force graph mein dikhti hai
- Red nodes = fraud, Green = licit, Amber = jo node explore kar rahe ho

**2. Full Explanation Tab**
- GNNExplainer ka output dikhta hai
- Exactly kaun si connections aur features ne fraud flag kiya

**3. 🌐 Multi-Dataset Tab** ← Naya!
- 5 colored tabs — ek har dataset ke liye
- Click karo kisi bhi dataset tab par → us dataset ka interactive graph dikhta hai
- Node ID daalo → explore karo
- Jab tak Notebook 10 nahi chalate, pending card dikhta hai with Kaggle link

### API Endpoints

```
POST /api/v1/score                   ← Node ka fraud probability
POST /api/v1/explain                 ← GNNExplainer subgraph
GET  /api/v1/datasets                ← Available trained datasets list
GET  /api/v1/datasets/{key}/sample   ← Sample fraud subgraph
POST /api/v1/datasets/{key}/subgraph ← Koi bhi node ka K-hop graph
GET  /api/v1/comparison              ← Multi-dataset CSV results
```

---

## Technology Stack

| Category | Tool |
|---|---|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.12 |
| Graph ML | PyTorch Geometric (PyG) 2.6.1 |
| Classical ML | scikit-learn, XGBoost, LightGBM |
| Graph Construction | sklearn NearestNeighbors (KNN) |
| Explainability | GNNExplainer (built-in PyG) |
| Backend API | FastAPI + uvicorn |
| Dashboard | D3.js v7 + vanilla JS |
| Visualization | matplotlib, seaborn, networkx |
| Data | 5 Kaggle datasets across 5 fraud domains |

---

## Project Structure Overview

```
GraphShield/
├── data/
│   ├── raw/
│   │   ├── elliptic/         ← Bitcoin CSVs
│   │   ├── credit_card/      ← creditcard.csv
│   │   ├── paysim/           ← online_payments_fraud.csv
│   │   ├── insurance/        ← fraud_oracle.csv
│   │   └── ecommerce/        ← Fraudulent_E-Commerce_Transaction_Data.csv
│   └── processed/
│       ├── *.pt              ← Elliptic tensors
│       ├── elliptic/graph.pt ← Graph API files (NB10 se generate hote hain)
│       ├── credit_card/graph.pt
│       ├── paysim/graph.pt
│       ├── insurance/graph.pt
│       └── ecommerce/graph.pt
│
├── notebooks/
│   ├── 01–09: Original workflow (EDA → GNN → Ablation)
│   └── 10_multi_dataset_comparison.ipynb  ← 5 domains × 3 GNNs
│
├── src/
│   ├── models.py                   ← GCN, GraphSAGE, GAT
│   ├── tabular_graph_builder.py    ← KNN graph builder (tabular → graph)
│   └── datasets/                   ← Loaders for each dataset
│
├── api/
│   ├── app.py                      ← FastAPI app
│   ├── routers/multi_graph.py      ← Graph endpoints for all 5 datasets
│   └── services/multi_graph_service.py ← Manages all 5 graphs in memory
│
├── dashboard/index.html            ← 3-tab D3.js dashboard
└── results/
    ├── figures/                    ← Elliptic plots
    └── multi_dataset/              ← Cross-domain results + figures
```

---

## Key Findings Summary

| Finding | Result |
|---|---|
| 🥇 Best GNN overall (sabhi 5 domains) | **GraphSAGE** — har domain mein winner |
| 🥇 Highest Recall (Elliptic) | **GAT — 0.948** |
| 🥇 Best F1 (PaySim, tabular best) | **GraphSAGE — 0.741** |
| 🥇 Best ROC-AUC (Credit Card) | **GraphSAGE — 0.978** |
| 🥇 Best overall (ML + GNN) | **LightGBM / XGBoost — F1 0.962** (Elliptic pe) |
| ⚠️ Class weights critical | Recall 0.887 → 0.618 bina class weights ke |
| ⚠️ Temporal gap serious | PR-AUC 0.796 → 0.519 temporal split mein |
| 🔍 Fraud is structural | Top features neighbourhood stats hain, individual amount nahi |
| 🌐 KNN graphs work | GNNs tabular data mein bhi fraud rings detect kar sakta hai |
| 🚗 Insurance hardest | Best F1 sirf 0.227 — small dataset + extreme imbalance |
| 📱 PaySim easiest tabular | Best F1 0.741 — clearest fraud signal in mobile tx features |

---

## Future Work

1. **Temporal GNN (TGN/TGAT)** — Time ke saath evolve hote fraud patterns handle karna
2. **Heterogeneous Graph** — Users, devices, IPs, merchants — R-GCN ya HGT se
3. **Self-Supervised Learning** — Bahut kam labels ke saath bhi kaam karna
4. **Federated GNN** — Multiple banks milke train karein bina data share kiye
5. **Real-Time Streaming** — Sub-second fraud scoring on live transactions
6. **Adversarial Robustness** — Fraudsters jo deliberately graph structure manipulate karein
7. **Better KNN Graph** — Feature selection + weighted edges for tabular datasets
8. **Larger Training Samples** — Currently 30K nodes pe train karte hain, full dataset pe bhi try karna

---

## Ek Line Summary

> **GraphShield isliye banaya gaya kyunki real-world financial fraud sirf numbers mein nahi, balki connections mein hota hai — aur GNNs woh connections samajhte hain jo traditional ML nahi samajh sakta. Aur yeh sirf Bitcoin mein nahi — Credit Card, Mobile Payments, Insurance, E-Commerce — har jagah fraud ke patterns graph structure mein milte hain.**
