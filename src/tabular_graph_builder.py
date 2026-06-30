"""
Build PyTorch Geometric graphs from tabular (feature_matrix, labels) data
using K-Nearest Neighbours (KNN) to define edges.

Why KNN graphs for tabular fraud data?
  - Transactions with similar feature profiles cluster together.
  - GNNs propagate fraud signals inside these clusters.
  - A clean-looking transaction surrounded by fraudulent neighbours
    gets a higher fraud score — exactly the "fraud ring" effect.
"""

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


def build_knn_edge_index(X: np.ndarray, k: int = 10, metric: str = "euclidean") -> np.ndarray:
    """
    Construct an undirected KNN edge list from feature matrix X.

    Returns
    -------
    edge_array : np.ndarray  shape (E, 2)  int64
        Each row is (src, dst).  Edges are deduplicated and bidirectional.
    """
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric=metric, n_jobs=-1)
    nbrs.fit(X)
    _, indices = nbrs.kneighbors(X)  # shape (N, k+1)

    edges = set()
    for i, neighbours in enumerate(indices):
        for j in neighbours[1:]:  # skip self (index 0)
            if i != j:
                edges.add((min(i, j), max(i, j)))  # canonical order for dedup

    # Make bidirectional
    edge_list = []
    for u, v in edges:
        edge_list.append((u, v))
        edge_list.append((v, u))

    return np.array(edge_list, dtype=np.int64)


def tabular_to_pyg(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Data:
    """
    Convert a tabular dataset to a PyTorch Geometric Data object.

    Steps
    -----
    1. Build KNN graph from X.
    2. Create PyG Data with node features and edges.
    3. Add stratified train / val / test masks.

    Parameters
    ----------
    X : np.ndarray  (N, F)  already normalised feature matrix
    y : np.ndarray  (N,)    integer class labels (0 = benign, 1 = fraud)
    k : int         number of neighbours per node
    metric : str    distance metric for NearestNeighbors
    test_size, val_size : float  fractions for splits
    random_state : int

    Returns
    -------
    data : torch_geometric.data.Data
    """
    N = len(X)
    print(f"  Building KNN graph (k={k}, metric={metric}) for {N:,} nodes × {X.shape[1]} features ...")
    edge_array = build_knn_edge_index(X, k=k, metric=metric)

    node_features = torch.tensor(X, dtype=torch.float)
    edge_index = torch.tensor(edge_array, dtype=torch.long).t().contiguous()
    labels = torch.tensor(y, dtype=torch.long)

    data = Data(x=node_features, edge_index=edge_index, y=labels)
    print(f"  Graph created  →  {data.num_nodes:,} nodes | {data.num_edges:,} edges")

    # Stratified splits
    indices = np.arange(N)
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, stratify=y[train_idx], random_state=random_state
    )

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    data.train_mask = train_mask
    data.val_mask   = val_mask
    data.test_mask  = test_mask

    fraud_pct = y.sum() / len(y) * 100
    print(f"  Fraud rate: {fraud_pct:.2f}%  |  Split: {train_mask.sum()} / {val_mask.sum()} / {test_mask.sum()}")
    return data
