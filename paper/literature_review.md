# Literature Review

## Traditional ML for Fraud Detection

Early fraud detection systems relied on rule-based engines. Machine learning replaced hard-coded rules with data-driven classifiers. [Dal Pozzolo et al., 2015] demonstrated that Random Forests with cost-sensitive learning outperform simpler classifiers on credit card fraud. XGBoost and LightGBM became the industry standard for tabular fraud datasets due to their handling of missing values and gradient-boosted ensemble strength.

A persistent challenge is class imbalance — fraud events constitute less than 1% of transactions in real datasets. Techniques such as SMOTE, cost-sensitive learning, and focal loss have been proposed to address this [Chawla et al., 2002].

## Graph-Based Approaches

The insight that fraud is often network-based motivated the application of graph analytics to financial data. [Akoglu et al., 2015] survey graph-based anomaly detection methods, highlighting guilt-by-association: nodes connected to flagged entities are themselves suspicious.

### Graph Neural Networks

[Kipf & Welling, 2017] introduced the Graph Convolutional Network (GCN), enabling semi-supervised node classification by aggregating neighbourhood features through spectral convolutions. [Hamilton et al., 2017] proposed GraphSAGE, an inductive variant that samples and aggregates from local neighbourhoods, scaling to graphs with millions of nodes. [Veličković et al., 2018] introduced Graph Attention Networks (GAT), replacing uniform aggregation with learned attention weights that identify the most relevant neighbours.

### GNNs for Fraud Detection

[Weber et al., 2019] applied GCN and GraphSAGE to the Elliptic Bitcoin dataset, showing that GNNs outperform non-graph baselines on illicit node classification. [Liu et al., 2021] propose Pick-and-Choose GNN for credit card fraud detection on heterogeneous graphs. [Dou et al., 2020] address the camouflage challenge in fraud detection with a graph-based model that handles inconsistencies between local features and neighbourhood structure.

## Explainability in GNNs

[Ying et al., 2019] introduced GNNExplainer, a post-hoc explanation method that identifies the minimal subgraph and feature subset important to a GNN's prediction. [Luo et al., 2020] propose PGExplainer for generating global explanations. Explainability is critical in financial fraud detection for regulatory compliance and investigator trust.
