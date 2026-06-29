import torch
import torch.nn.functional as F
from fastapi import APIRouter, HTTPException
from api.schemas import ExplainRequest, ExplainResponse, EdgeInfo
from api.services.model_service import get_model_service
from api.services.graph_service import get_graph_service
from api.services.explain_service import build_explanation

router = APIRouter(prefix="/explain", tags=["Explainability"])


@router.post("/", response_model=ExplainResponse, summary="Explain a fraud prediction")
def explain_node(req: ExplainRequest):
    ms = get_model_service()
    gs = get_graph_service()

    if not ms.loaded or not gs.loaded:
        raise HTTPException(503, "Services not ready")

    if req.node_id < 0 or req.node_id >= gs.num_nodes:
        raise HTTPException(400, f"node_id must be in [0, {gs.num_nodes - 1}]")

    x          = torch.tensor(gs._features, dtype=torch.float32)
    edge_index = torch.tensor(gs._edge_index, dtype=torch.long)

    with torch.no_grad():
        logits     = ms.model(x, edge_index)
        probs      = F.softmax(logits, dim=-1).numpy()
        fraud_prob = float(probs[req.node_id, 1])

    sub_nodes = gs.k_hop_subgraph(req.node_id, k=2, max_nodes=80)
    sub_edges = gs.subgraph_edges(sub_nodes)
    feat      = gs.get_features(req.node_id)

    exp = build_explanation(
        node_id        = req.node_id,
        subgraph_nodes = sub_nodes,
        subgraph_edges = sub_edges,
        node_features  = feat,
        labels         = gs._labels,
        fraud_probs    = probs[:, 1],
        top_k_edges    = req.top_k_edges,
    )

    return ExplainResponse(
        node_id           = req.node_id,
        fraud_probability = round(fraud_prob, 6),
        explanation_edges = [EdgeInfo(**e) for e in exp["edges"]],
        explanation_nodes = exp["nodes"],
        top_features      = exp["top_features"],
        summary           = exp.get("summary", ""),
    )
