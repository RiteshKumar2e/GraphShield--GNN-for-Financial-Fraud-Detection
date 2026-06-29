import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_elliptic_data(feature_path, class_path, edge_path):
    features = pd.read_csv(feature_path, header=None)
    classes = pd.read_csv(class_path)
    edges = pd.read_csv(edge_path)
    return features, classes, edges


def clean_labels(classes):
    classes = classes.copy()
    classes = classes[classes["class"] != "unknown"]
    classes["label"] = classes["class"].map({"1": 1, "2": 0, 1: 1, 2: 0})
    return classes


def align_features_with_labels(features, classes):
    """Keep only nodes that have known labels."""
    known_ids = set(classes.iloc[:, 0].values)
    node_id_col = features.iloc[:, 0]
    mask = node_id_col.isin(known_ids)
    features_filtered = features[mask].reset_index(drop=True)
    return features_filtered


def normalize_features(feature_matrix: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(feature_matrix)


def build_node_id_map(node_ids):
    """Map raw node IDs to consecutive integer indices."""
    return {nid: idx for idx, nid in enumerate(node_ids)}


def remap_edges(edges, node_id_map):
    """Convert edge list from raw IDs to integer indices."""
    src = edges.iloc[:, 0].map(node_id_map).dropna().astype(int)
    dst = edges.iloc[:, 1].map(node_id_map).dropna().astype(int)
    valid = src.notna() & dst.notna()
    return src[valid].values, dst[valid].values


def prepare_elliptic(feature_path, class_path, edge_path):
    """Full pipeline: load → clean → align → normalize → remap edges."""
    features, classes, edges = load_elliptic_data(feature_path, class_path, edge_path)
    classes = clean_labels(classes)
    features = align_features_with_labels(features, classes)

    node_ids = features.iloc[:, 0].values
    node_id_map = build_node_id_map(node_ids)

    feature_matrix = features.iloc[:, 1:].values.astype(float)
    feature_matrix = normalize_features(feature_matrix)

    labels = classes.set_index(classes.columns[0])["label"].reindex(node_ids).values

    src, dst = remap_edges(edges, node_id_map)
    edge_index_array = np.stack([src, dst], axis=1)

    return feature_matrix, edge_index_array, labels
