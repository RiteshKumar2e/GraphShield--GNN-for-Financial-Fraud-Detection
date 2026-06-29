# Conclusion and Future Work

## Conclusion

This paper presents **GraphShield**, an explainable graph neural network framework for financial fraud detection evaluated on the Elliptic Bitcoin dataset. The key experimental findings are:

1. **GNNs maximise recall.** GAT achieves the highest recall (0.9516) of all models, surpassing even XGBoost (0.9329). For fraud detection — where missing a fraud case is costlier than a false alarm — GNNs offer a meaningful advantage.

2. **Tree models dominate F1 when features include pre-aggregated graph statistics.** XGBoost (F1 0.9620) and LightGBM (F1 0.9616) outperform GNNs on F1 because the Elliptic node features already embed 72 neighbourhood-aggregated statistics. GNNs re-derive this structure from raw edges; their advantage is strongest in settings without such feature engineering.

3. **GraphSAGE is the best-balanced GNN.** It achieves F1 0.7321, ROC-AUC 0.9770, and PR-AUC 0.8995 — the strongest overall GNN result.

4. **Class weighting is critical.** Removing class weights causes recall to drop from 0.89 to 0.62 on the fraud class, demonstrating that naive training on imbalanced data severely under-detects fraud.

5. **Temporal generalisation is hard.** Under a temporal split (train on time steps 1–4, test on 5–49), GCN's PR-AUC falls from 0.7956 to 0.5185 — confirming that fraud patterns shift over time and that temporal evaluation is a more rigorous benchmark than random splitting.

6. **GNNExplainer surfaces fraud rings.** For the highest-confidence fraud node (fraud prob. 0.9994), GNNExplainer identified a dense subgraph of illicit neighbours as the primary evidence — consistent with known fraud ring behaviour. Top predictive features are neighbourhood-aggregated statistics rather than individual transaction amounts.

---

## Future Work

1. **Temporal GNN:** Apply TGN or TGAT to explicitly model how fraud patterns evolve across the 49 time steps. The temporal split experiment (PR-AUC 0.5185) motivates this strongly.

2. **Heterogeneous Graph:** Incorporate multiple entity types (user, device, IP, merchant) as distinct node types with typed edges — using R-GCN or HGT — for multi-entity fraud detection beyond transaction-only graphs.

3. **Self-Supervised Pre-training:** Use contrastive graph learning (GraphCL, GRACE) to reduce label dependency in settings where fraud labels are scarce or delayed.

4. **Federated GNN:** Extend to a multi-bank setting where each institution trains a local GNN and shares only model gradients, enabling cross-institution fraud detection while preserving data privacy.

5. **Real-Time Detection:** Integrate with a streaming transaction pipeline using dynamic graph updates and online GNN inference for sub-second fraud scoring.

6. **Adversarial Robustness:** Evaluate model robustness when fraudsters deliberately camouflage graph connectivity to evade GNN-based detection.

---

## References

- Weber, M. et al. (2019). Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics. SIGKDD Workshop on Anomaly Detection in Finance.
- Kipf, T. N. & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.
- Hamilton, W. et al. (2017). Inductive Representation Learning on Large Graphs. NeurIPS.
- Veličković, P. et al. (2018). Graph Attention Networks. ICLR.
- Ying, R. et al. (2019). GNNExplainer: Generating Explanations for Graph Neural Networks. NeurIPS.
- Dou, Y. et al. (2020). Enhancing Graph Neural Network-based Fraud Detection via Locally Homophilous Edges. WWW.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.
