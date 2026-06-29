# Methodology

## 1. Dataset

**Elliptic Bitcoin Dataset** (Weber et al., 2019):
- 203,769 nodes (Bitcoin transactions)
- 234,355 edges (transaction flows)
- 166 node features (94 local + 72 aggregated neighbourhood features)
- Labels: illicit (class 1), licit (class 2), unknown (~79% unlabelled)
- 49 discrete time steps

**After preprocessing (labeled nodes only):**
- Nodes: 46,564 (41,941 licit + 4,545 illicit)
- Edges: 36,624
- Fraud rate: 9.77%
- Train / Val / Test: 29,800 / 7,451 / 9,313 (stratified split)

## 2. Graph Construction

Nodes represent Bitcoin transactions. A directed edge from node *u* to node *v* indicates that transaction *u* is an input to transaction *v*. Node features include transaction amount statistics, time step, and aggregated one-hop neighbourhood statistics.

The raw dataset is converted to a `torch_geometric.data.Data` object:

```
x          ∈ ℝ^(N × 166)     node feature matrix
edge_index ∈ ℤ^(2 × E)       directed adjacency in COO format
y          ∈ {0, 1}^N         binary fraud label
```

## 3. Train / Validation / Test Split

A stratified random split is applied to labelled nodes:
- Train: 64%
- Validation: 16%
- Test: 20%

A temporal split (train on time steps 1–34, test on 35–49) is used in the ablation study to simulate realistic deployment conditions.

## 4. Class Imbalance Handling

The fraud class represents approximately 9% of labelled nodes. Inverse-frequency class weights are computed from training labels and applied to the cross-entropy loss:

```
w_c = 1 / count(c),  normalised so Σ w_c = num_classes
```

## 5. Model Architectures

### GCN
Two GCNConv layers with ReLU activation and dropout (p=0.4) between layers. Output dimension = 2 (binary classification).

### GraphSAGE
Two SAGEConv layers with the same activation and dropout. Uses mean aggregation over sampled neighbours.

### GAT
Two GATConv layers. First layer: 4 attention heads × 32 hidden units. Second layer: single head, output dimension 2. ELU activation between layers.

### DeepGCN (ablation)
Three GCNConv layers with BatchNorm after each intermediate layer.

## 6. Training

All GNN models are trained with:
- Optimiser: Adam (lr=0.001, weight_decay=5×10⁻⁴)
- Epochs: 200
- Loss: weighted cross-entropy
- Early stopping criterion: monitored validation loss

## 7. Evaluation Metrics

Given the class imbalance, accuracy is not the primary metric:

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall    | TP / (TP + FN) |
| F1-score  | 2 × P × R / (P + R) |
| ROC-AUC   | Area under the ROC curve |
| PR-AUC    | Area under the Precision-Recall curve |

PR-AUC is the most informative metric for heavily imbalanced datasets.

## 8. Explainability

GNNExplainer (Ying et al., 2019) optimises a soft mask over edges and node features to identify the minimal subgraph that maximises mutual information with the GNN's prediction. The explanation is visualised as a subgraph highlighting the most suspicious transaction connections.
