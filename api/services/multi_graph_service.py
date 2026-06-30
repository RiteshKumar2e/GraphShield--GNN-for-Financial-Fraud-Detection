"""
Manages multiple dataset graphs in memory for the Multi-Dataset panel.
After Notebook 10 runs, each dataset saves a graph.pt to data/processed/{key}/.
This service loads all available graphs and serves subgraph queries.

Compatible with Python 3.8+.
"""
from __future__ import annotations

import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DATASET_ORDER = ['elliptic', 'credit_card', 'paysim', 'insurance', 'ecommerce']

DATASET_META = {
    'elliptic':    {'name': 'Elliptic Bitcoin',  'domain': 'Blockchain / Crypto',       'icon': '₿',  'color': '#f97316', 'kaggle': 'https://www.kaggle.com/datasets/ellipticco/elliptic-data-set'},
    'credit_card': {'name': 'Credit Card',        'domain': 'Banking / Credit Card',     'icon': '💳', 'color': '#1d4ed8', 'kaggle': 'https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud'},
    'paysim':      {'name': 'PaySim Payments',    'domain': 'Mobile Payments / Fintech', 'icon': '📱', 'color': '#7c3aed', 'kaggle': 'https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset'},
    'insurance':   {'name': 'Vehicle Insurance',  'domain': 'Vehicle Insurance Claims',  'icon': '🚗', 'color': '#16a34a', 'kaggle': 'https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection'},
    'ecommerce':   {'name': 'E-Commerce',         'domain': 'Online Retail',             'icon': '🛒', 'color': '#dc2626', 'kaggle': 'https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions'},
}


class MultiGraphService:

    def __init__(self, processed_dir: str = 'data/processed'):
        self._base: Path = Path(processed_dir)
        self._graphs: Dict[str, dict] = {}
        self.loaded: bool = False

    # ── Load ────────────────────────────────────────────────────

    def load(self) -> None:
        self._try_load_elliptic()
        for key in ['credit_card', 'paysim', 'insurance', 'ecommerce']:
            self._try_load_saved(key)
        self.loaded = True
        loaded_names = [DATASET_META[k]['name'] for k in self._graphs]
        print(f"  [MultiGraph] Loaded: {loaded_names if loaded_names else 'none — run Notebook 10 for tabular datasets'}")

    def _try_load_elliptic(self) -> None:
        ei_path = self._base / 'edge_index.pt'
        y_path  = self._base / 'labels.pt'
        if not (ei_path.exists() and y_path.exists()):
            print("  [MultiGraph] elliptic: processed files not found (run Notebook 02)")
            return
        try:
            edge_index = torch.load(ei_path, weights_only=True)
            y          = torch.load(y_path,  weights_only=True)

            # Better fraud probs saved by NB10 (all nodes); fall back to binary labels
            fp_path = self._base / 'elliptic' / 'graph.pt'
            fraud_probs: np.ndarray = y.float().numpy()
            if fp_path.exists():
                saved = torch.load(fp_path, weights_only=True)
                if 'fraud_probs' in saved:
                    fraud_probs = saved['fraud_probs'].numpy()

            self._graphs['elliptic'] = {
                'edge_index':  edge_index,
                'y':           y,
                'adj':         self._build_adj(edge_index),
                'fraud_probs': fraud_probs,
                'directed':    True,
            }
        except Exception as exc:
            print(f"  [MultiGraph] elliptic load error: {exc}")

    def _try_load_saved(self, key: str) -> None:
        path = self._base / key / 'graph.pt'
        if not path.exists():
            return
        try:
            saved = torch.load(path, weights_only=True)
            edge_index  = saved['edge_index']
            y           = saved['y']
            fraud_probs = saved.get('fraud_probs', y.float()).numpy()
            self._graphs[key] = {
                'edge_index':  edge_index,
                'y':           y,
                'adj':         self._build_adj(edge_index),
                'fraud_probs': fraud_probs,
                'directed':    False,
            }
        except Exception as exc:
            print(f"  [MultiGraph] {key} load error: {exc}")

    @staticmethod
    def _build_adj(edge_index: torch.Tensor) -> Dict[int, set]:
        # Bidirectional so BFS works for directed graphs too
        # (a node with only incoming edges would return 0 neighbours otherwise)
        adj: Dict[int, set] = defaultdict(set)
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        for s, d in zip(src, dst):
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
        return adj

    # ── Query helpers ────────────────────────────────────────────

    def available_keys(self) -> List[str]:
        return [k for k in DATASET_ORDER if k in self._graphs]

    def get_stats(self, key: str) -> Optional[dict]:
        g = self._graphs.get(key)
        if g is None:
            return None
        y    = g['y'].numpy()
        meta = DATASET_META.get(key, {})
        return {
            'key':        key,
            'name':       meta.get('name', key),
            'domain':     meta.get('domain', ''),
            'icon':       meta.get('icon', '📊'),
            'color':      meta.get('color', '#6b7280'),
            'kaggle':     meta.get('kaggle', ''),
            'num_nodes':  int(y.shape[0]),
            'num_edges':  int(g['edge_index'].shape[1]),
            'num_fraud':  int((y == 1).sum()),
            'num_legit':  int((y == 0).sum()),
            'fraud_rate': round(float((y == 1).mean() * 100), 2),
            'directed':   bool(g.get('directed', False)),
        }

    def get_sample_node(self, key: str) -> int:
        """Return the node with highest fraud probability for initial display."""
        g = self._graphs.get(key)
        if g is None:
            return 0
        y  = g['y'].numpy()
        fp = g['fraud_probs']
        fraud_idx = np.where(y == 1)[0]
        if len(fraud_idx) == 0:
            return 0
        return int(fraud_idx[np.argmax(fp[fraud_idx])])

    def k_hop_subgraph(self, key: str, node_id: int,
                       k: int = 2, max_nodes: int = 40) -> Tuple[List[dict], List[dict]]:
        g = self._graphs.get(key)
        if g is None:
            return [], []

        adj = g['adj']
        y   = g['y'].numpy()
        fp  = g['fraud_probs']

        # BFS to collect k-hop neighbourhood
        visited:  set = {node_id}
        frontier: set = {node_id}
        for _ in range(k):
            nxt: set = set()
            for n in frontier:
                for nb in adj.get(n, set()):
                    if nb not in visited:
                        nxt.add(nb)
            frontier = nxt
            visited |= frontier
            if len(visited) >= max_nodes:
                break

        visited_list = list(visited)[:max_nodes]
        visited_set  = set(visited_list)

        nodes = [{
            'id':         nid,
            'label':      int(y[nid]),
            'fraud_prob': round(float(fp[nid]), 4),
            'is_target':  nid == node_id,
        } for nid in visited_list]

        ei        = g['edge_index'].numpy()
        seen: set = set()
        edges: List[dict] = []
        for s, d in zip(ei[0], ei[1]):
            s, d = int(s), int(d)
            if s in visited_set and d in visited_set:
                canonical = (min(s, d), max(s, d))
                if canonical not in seen:
                    seen.add(canonical)
                    edges.append({'source': s, 'target': d, 'weight': 0.5})

        return nodes, edges


_instance: Optional[MultiGraphService] = None


def get_multi_graph_service() -> MultiGraphService:
    global _instance
    if _instance is None:
        _instance = MultiGraphService()
    return _instance
