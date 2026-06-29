"""
In-memory graph store backed by the processed PyG tensors.
Drop-in replaceable with Neo4j by swapping this module.
"""

import torch
import numpy as np
from collections import deque
from pathlib import Path


class GraphService:
    def __init__(self, processed_dir: str = "data/processed"):
        self._edge_index: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._features: np.ndarray | None = None
        self._adj: dict[int, set[int]] = {}
        self._processed_dir = Path(processed_dir)
        self.loaded = False

    def load(self):
        ei  = torch.load(self._processed_dir / "edge_index.pt", weights_only=True)
        lbl = torch.load(self._processed_dir / "labels.pt", weights_only=True)
        ft  = torch.load(self._processed_dir / "node_features.pt", weights_only=True)

        self._edge_index = ei.numpy()
        self._labels     = lbl.numpy()
        self._features   = ft.numpy()

        src, dst = self._edge_index[0], self._edge_index[1]
        for s, d in zip(src, dst):
            self._adj.setdefault(int(s), set()).add(int(d))
            self._adj.setdefault(int(d), set()).add(int(s))

        self.loaded = True

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def num_nodes(self) -> int:
        return int(self._labels.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self._edge_index.shape[1])

    @property
    def num_illicit(self) -> int:
        return int((self._labels == 1).sum())

    @property
    def num_licit(self) -> int:
        return int((self._labels == 0).sum())

    @property
    def avg_degree(self) -> float:
        return round(self.num_edges * 2 / self.num_nodes, 2)

    def get_label(self, node_id: int) -> int:
        return int(self._labels[node_id])

    def get_features(self, node_id: int) -> np.ndarray:
        return self._features[node_id]

    def k_hop_subgraph(self, node_id: int, k: int = 2, max_nodes: int = 80) -> set[int]:
        visited = {node_id}
        queue   = deque([(node_id, 0)])
        while queue and len(visited) < max_nodes:
            n, depth = queue.popleft()
            if depth >= k:
                continue
            for nb in self._adj.get(n, []):
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, depth + 1))
                    if len(visited) >= max_nodes:
                        break
        return visited

    def subgraph_edges(self, nodes: set[int]) -> list[tuple[int, int]]:
        src, dst = self._edge_index[0], self._edge_index[1]
        return [(int(s), int(d)) for s, d in zip(src, dst)
                if int(s) in nodes and int(d) in nodes]

    def node_info(self, node_id: int) -> dict:
        return {
            "id":    node_id,
            "label": int(self._labels[node_id]),
            "degree": len(self._adj.get(node_id, [])),
        }


# module-level singleton
_graph_service: GraphService | None = None


def get_graph_service() -> GraphService:
    global _graph_service
    if _graph_service is None:
        _graph_service = GraphService()
    return _graph_service
