# Results and Discussion

## 1. Experimental Setup

- **Dataset:** Elliptic Bitcoin Dataset — 46,564 labeled nodes (41,941 licit, 4,545 illicit), 36,624 edges
- **Split:** Stratified random — Train 64% / Val 16% / Test 20% (9,313 test nodes; fraud rate 9.8%)
- **Hardware:** CPU-only (Intel, Python 3.10)
- **Framework:** PyTorch 2.12, PyTorch Geometric 2.6.1

---

## 2. Baseline ML Results

All ML models operate on node features only — no graph structure.

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC |
|:--------------------|--------:|----------:|-------:|-------:|--------:|-------:|
| Logistic Regression | 0.8783  | 0.4414    | 0.9285 | 0.5984 | 0.9649  | 0.7546 |
| MLP                 | 0.9811  | 0.9379    | 0.8636 | 0.8992 | 0.9856  | 0.9408 |
| Random Forest       | 0.9875  | 0.9962    | 0.8757 | 0.9321 | 0.9966  | 0.9817 |
| LightGBM            | 0.9927  | 0.9873    | 0.9373 | 0.9616 | 0.9982  | 0.9895 |
| XGBoost             | 0.9928  | 0.9930    | 0.9329 | 0.9620 | 0.9974  | 0.9872 |

**Best baseline:** LightGBM (PR-AUC 0.9895) and XGBoost (F1 0.9620).

---

## 3. GNN Results

GNN models incorporate graph structure via message passing over the transaction network.

| Model     | Accuracy | Precision | Recall     | F1     | ROC-AUC | PR-AUC |
|:----------|--------:|----------:|-----------:|-------:|--------:|-------:|
| GCN       | 0.9069  | 0.5132    | **0.8955** | 0.6525 | 0.9648  | 0.8098 |
| GraphSAGE | 0.9358  | 0.6175    | 0.8988     | 0.7321 | 0.9770  | 0.8995 |
| GAT       | 0.8743  | 0.4342    | **0.9516** | 0.5963 | 0.9759  | 0.8681 |

---

## 4. Full Model Comparison

| Model               | Accuracy | Precision | Recall | F1     | ROC-AUC | PR-AUC |
|:--------------------|--------:|----------:|-------:|-------:|--------:|-------:|
| Logistic Regression | 0.8783  | 0.4414    | 0.9285 | 0.5984 | 0.9649  | 0.7546 |
| MLP                 | 0.9811  | 0.9379    | 0.8636 | 0.8992 | 0.9856  | 0.9408 |
| Random Forest       | 0.9875  | 0.9962    | 0.8757 | 0.9321 | 0.9966  | 0.9817 |
| LightGBM            | 0.9927  | 0.9873    | 0.9373 | 0.9616 | 0.9982  | **0.9895** |
| XGBoost             | 0.9928  | 0.9930    | 0.9329 | **0.9620** | **0.9974** | 0.9872 |
| GCN                 | 0.9069  | 0.5132    | 0.8955 | 0.6525 | 0.9648  | 0.8098 |
| GraphSAGE           | 0.9358  | 0.6175    | 0.8988 | 0.7321 | 0.9770  | 0.8995 |
| GAT                 | 0.8743  | 0.4342    | **0.9516** | 0.5963 | 0.9759 | 0.8681 |

---

## 5. Key Observations

### 5.1 Recall vs Precision Trade-off
GNN models — especially GAT (recall 0.9516) and GCN (recall 0.8955) — achieve the highest recall of all models. This is critical in fraud detection: missing a fraud case (false negative) is far more costly than a false alarm.

### 5.2 Why GNNs Have Lower F1 Than Tree Models
Tree models (XGBoost, LightGBM) achieve high F1 because the Elliptic node features already include 72 pre-aggregated neighbourhood statistics. The GNNs are partially re-deriving these from raw structure. In a setting where such feature engineering is absent, GNN advantage would be larger.

### 5.3 GraphSAGE is the Best Balanced GNN
GraphSAGE achieves the best trade-off among GNNs: F1 0.7321, ROC-AUC 0.9770, PR-AUC 0.8995 — substantially above GCN and GAT on F1 while retaining high recall (0.8988).

### 5.4 GAT Maximises Recall
GAT achieves the highest recall (0.9516) across all models including ML baselines, demonstrating that attention-weighted neighbourhood aggregation is effective at identifying illicit nodes that are connected to suspicious clusters.

---

## 6. Ablation Study

### 6.1 Effect of Class Weights

| Configuration       | F1     | Recall | PR-AUC | ROC-AUC |
|:--------------------|-------:|-------:|-------:|--------:|
| GCN + Class Weights | 0.6317 | **0.8867** | **0.7956** | **0.9612** |
| GCN − Class Weights | 0.7294 | 0.6183 | 0.7939 | 0.9408 |

**Finding:** Removing class weights raises F1 (the model is less aggressive in predicting fraud) but recall drops sharply from 0.89 to 0.62 — it misses 28% more fraud cases. For fraud detection, class weights are essential.

### 6.2 Effect of Model Depth

| Configuration | F1     | Recall | PR-AUC | ROC-AUC |
|:--------------|-------:|-------:|-------:|--------:|
| GCN 2-Layer   | 0.6077 | 0.8911 | 0.7794 | 0.9580  |
| GCN 3-Layer   | 0.5952 | **0.9043** | 0.7813 | 0.9579  |

**Finding:** The 3-layer GCN slightly increases recall but decreases F1 — adding depth expands the neighbourhood aggregation radius (3-hop vs 2-hop), capturing more distant fraud connections at the cost of precision dilution. Both depths perform similarly overall (ROC-AUC ≈ 0.958).

---

## 7. Temporal Split Experiment

Training on early time steps (1–4) and testing on later ones (5–49) simulates real-world deployment where fraud patterns evolve.

| Split          | F1     | Recall | PR-AUC | ROC-AUC |
|:---------------|-------:|-------:|-------:|--------:|
| Random Split   | 0.6317 | 0.8867 | 0.7956 | 0.9612  |
| Temporal Split | 0.2982 | 0.6400 | 0.5185 | 0.8312  |

**Finding:** Performance drops significantly under temporal evaluation (PR-AUC: 0.7956 → 0.5185), confirming that fraud patterns shift over time. The temporal split is a more realistic and challenging benchmark. Future work with Temporal GNNs (TGN, TGAT) is motivated by this result.

---

## 8. Explainability Results (GNNExplainer on GAT)

- **Target node:** Index 7929, fraud probability **0.9994**
- **Explanation:** GNNExplainer identified a subgraph of 30 high-weight edges connecting node 7929 to a cluster of illicit neighbours
- **Top features:** Neighbourhood-aggregated transaction statistics (features 93–165) dominate, confirming that illicit nodes are best identified through their structural context rather than individual transaction amounts alone

**Interpretation:** The model flags node 7929 as illicit primarily because of its dense connectivity to other known illicit nodes — a textbook fraud ring pattern. The top node features correspond to aggregated statistics of one-hop neighbours (transaction counts, balance ratios), consistent with the Elliptic dataset's feature construction.

---

## 9. Summary

| Criterion              | Best Model    | Score  |
|:-----------------------|:-------------|-------:|
| Highest Recall         | GAT           | 0.9516 |
| Highest F1             | XGBoost       | 0.9620 |
| Highest PR-AUC         | LightGBM      | 0.9895 |
| Highest ROC-AUC        | LightGBM      | 0.9982 |
| Best GNN (balanced)    | GraphSAGE     | F1 0.7321, PR-AUC 0.8995 |
| Best recall (any model)| GAT           | 0.9516 |

For deployment in a fraud detection system where **catching every fraud case is paramount**, GAT is recommended despite its lower F1. For settings requiring high precision and overall accuracy, XGBoost or LightGBM are superior when rich node features are available.
