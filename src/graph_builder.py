import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def build_pyg_graph(node_features: np.ndarray, edge_index_array: np.ndarray, labels: np.ndarray) -> Data:
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index_array, dtype=torch.long).t().contiguous()
    y = torch.tensor(labels, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, y=y)


def add_train_val_test_masks(data: Data, test_size=0.2, val_size=0.2, random_state=42) -> Data:
    num_nodes = data.num_nodes
    indices = np.arange(num_nodes)
    y = data.y.numpy()

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, stratify=y[train_idx], random_state=random_state
    )

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def save_graph(data: Data, save_dir: str):
    import os
    os.makedirs(save_dir, exist_ok=True)
    torch.save(data.x, f"{save_dir}/node_features.pt")
    torch.save(data.edge_index, f"{save_dir}/edge_index.pt")
    torch.save(data.y, f"{save_dir}/labels.pt")
    masks = {
        "train_mask": data.train_mask,
        "val_mask": data.val_mask,
        "test_mask": data.test_mask,
    }
    torch.save(masks, f"{save_dir}/train_val_test_masks.pt")
    print(f"Graph saved to {save_dir}/")


def load_graph(save_dir: str) -> Data:
    x = torch.load(f"{save_dir}/node_features.pt")
    edge_index = torch.load(f"{save_dir}/edge_index.pt")
    y = torch.load(f"{save_dir}/labels.pt")
    masks = torch.load(f"{save_dir}/train_val_test_masks.pt")
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = masks["train_mask"]
    data.val_mask = masks["val_mask"]
    data.test_mask = masks["test_mask"]
    return data
