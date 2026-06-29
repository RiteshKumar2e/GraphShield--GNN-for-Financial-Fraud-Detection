import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer


def get_explainer(model, epochs=200):
    return Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )


def explain_node(explainer, data, node_idx):
    explanation = explainer(data.x, data.edge_index, index=node_idx)
    return explanation


def plot_explanation_subgraph(explanation, node_idx, title=None, save_path=None, top_k=30):
    import numpy as np
    edge_mask = explanation.edge_mask.cpu().numpy()
    edge_index = explanation.edge_index.cpu().numpy()

    # Keep only top-K edges by mask score to keep the graph renderable
    top_k = min(top_k, len(edge_mask))
    top_indices = np.argsort(edge_mask)[::-1][:top_k]

    G = nx.DiGraph()
    for i in top_indices:
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        G.add_edge(src, dst, weight=float(edge_mask[i]))

    if G.number_of_nodes() == 0:
        print(f"No important edges found for node {node_idx}.")
        return

    print(f"Explanation subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    node_colors = ["red" if n == node_idx else "lightblue" for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges()]

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42, iterations=50)
    nx.draw(G, pos, node_color=node_colors, width=edge_widths,
            with_labels=True, node_size=500, font_size=7, arrows=True)
    plt.title(title or f"Explanation Subgraph for Node {node_idx}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()


def top_k_node_features(explanation, feature_names=None, k=10):
    node_mask = explanation.node_mask.mean(dim=0).cpu().numpy()
    indices = node_mask.argsort()[::-1][:k]
    if feature_names is not None:
        top_features = [(feature_names[i], float(node_mask[i])) for i in indices]
    else:
        top_features = [(f"Feature_{i}", float(node_mask[i])) for i in indices]
    for name, score in top_features:
        print(f"  {name}: {score:.4f}")
    return top_features
