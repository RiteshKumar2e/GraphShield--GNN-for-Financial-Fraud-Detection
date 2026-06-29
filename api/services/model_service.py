"""
Loads the best trained GNN model and runs inference.
Tries GAT → GraphSAGE → GCN in preference order.
"""

import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from models import GAT, GraphSAGE, GCN


_MODEL_PRIORITY = [
    ("GAT",        "models/gat_model.pt",        GAT),
    ("GraphSAGE",  "models/graphsage_model.pt",   GraphSAGE),
    ("GCN",        "models/gcn_model.pt",         GCN),
]

_HIDDEN    = 64
_OUT       = 2
_IN_FEATS  = 166
_DROPOUT   = 0.4
_GAT_HEADS = 4


def _build_model(cls, name: str):
    if cls is GAT:
        return cls(_IN_FEATS, _HIDDEN // _GAT_HEADS, _OUT, heads=_GAT_HEADS, dropout=_DROPOUT)
    return cls(_IN_FEATS, _HIDDEN, _OUT, dropout=_DROPOUT)


class ModelService:
    def __init__(self):
        self.model      = None
        self.model_name = "none"
        self.device     = torch.device("cpu")
        self.loaded     = False
        self._predictions_served = 0

    def load(self):
        for name, path, cls in _MODEL_PRIORITY:
            p = Path(path)
            if not p.exists():
                continue
            try:
                model = _build_model(cls, name)
                state = torch.load(p, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                model.eval()
                self.model      = model
                self.model_name = name
                self.loaded     = True
                print(f"[ModelService] Loaded {name} from {path}")
                return
            except Exception as e:
                print(f"[ModelService] Failed to load {name}: {e}")
        raise RuntimeError("No trained model found in models/. Run the training notebooks first.")

    @property
    def predictions_served(self) -> int:
        return self._predictions_served

    @torch.no_grad()
    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Returns softmax probabilities [N, 2]."""
        logits = self.model(x, edge_index)
        probs  = F.softmax(logits, dim=-1)
        return probs.numpy()

    @torch.no_grad()
    def predict_node(
        self,
        node_id: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> dict:
        """Score a single node (full graph context, index into result)."""
        t0    = time.perf_counter()
        probs = self.predict(x, edge_index)
        self._predictions_served += 1

        fraud_prob = float(probs[node_id, 1])
        licit_prob = float(probs[node_id, 0])

        threshold = 0.5
        if fraud_prob >= 0.8:
            verdict, confidence = "ILLICIT", "HIGH"
        elif fraud_prob >= 0.5:
            verdict, confidence = "ILLICIT", "MEDIUM"
        elif fraud_prob >= 0.3:
            verdict, confidence = "REVIEW",  "LOW"
        else:
            verdict, confidence = "LICIT",   "HIGH" if licit_prob >= 0.8 else "MEDIUM"

        return {
            "fraud_probability": round(fraud_prob, 6),
            "licit_probability": round(licit_prob, 6),
            "verdict":           verdict,
            "confidence":        confidence,
            "threshold_used":    threshold,
            "inference_ms":      round((time.perf_counter() - t0) * 1000, 2),
        }


# module-level singleton
_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
