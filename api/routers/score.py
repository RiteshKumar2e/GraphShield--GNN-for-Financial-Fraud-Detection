import datetime
import torch
from fastapi import APIRouter, HTTPException, Depends
from api.schemas import ScoreRequest, ScoreResponse, EdgeInfo
from api.services.model_service import get_model_service, ModelService
from api.services.graph_service import get_graph_service, GraphService
from api.services.explain_service import top_features

router = APIRouter(prefix="/score", tags=["Scoring"])


def _get_services():
    ms = get_model_service()
    gs = get_graph_service()
    if not ms.loaded or not gs.loaded:
        raise HTTPException(503, "Services not ready yet")
    return ms, gs


@router.post("/", response_model=ScoreResponse, summary="Score a single transaction node")
def score_node(req: ScoreRequest):
    ms, gs = _get_services()

    if req.node_id < 0 or req.node_id >= gs.num_nodes:
        raise HTTPException(400, f"node_id must be in [0, {gs.num_nodes - 1}]")

    # Load full graph tensors (cached in graph service)
    x          = torch.tensor(gs._features, dtype=torch.float32)
    edge_index = torch.tensor(gs._edge_index, dtype=torch.long)

    result = ms.predict_node(req.node_id, x, edge_index)

    # Build subgraph
    sub_nodes = gs.k_hop_subgraph(req.node_id, k=1, max_nodes=40)
    sub_edges = gs.subgraph_edges(sub_nodes)

    feat = gs.get_features(req.node_id)

    return ScoreResponse(
        node_id           = req.node_id,
        fraud_probability = result["fraud_probability"],
        licit_probability = result["licit_probability"],
        verdict           = result["verdict"],
        confidence        = result["confidence"],
        threshold_used    = result["threshold_used"],
        top_features      = top_features(feat, top_k=5),
        subgraph_nodes    = list(sub_nodes),
        subgraph_edges    = [EdgeInfo(source=s, target=d, weight=1.0) for s, d in sub_edges],
        timestamp         = datetime.datetime.utcnow().isoformat() + "Z",
    )


@router.post("/batch", summary="Score multiple transaction nodes")
def score_batch(node_ids: list[int]):
    ms, gs = _get_services()

    if len(node_ids) > 500:
        raise HTTPException(400, "Maximum batch size is 500")

    x          = torch.tensor(gs._features, dtype=torch.float32)
    edge_index = torch.tensor(gs._edge_index, dtype=torch.long)

    import torch.nn.functional as F
    import torch
    with torch.no_grad():
        logits = ms.model(x, edge_index)
        probs  = F.softmax(logits, dim=-1).numpy()

    results = []
    for nid in node_ids:
        if nid < 0 or nid >= gs.num_nodes:
            continue
        fp = float(probs[nid, 1])
        results.append({
            "node_id":          nid,
            "fraud_probability": round(fp, 6),
            "verdict":          "ILLICIT" if fp >= 0.5 else "LICIT",
        })

    return {"results": results, "count": len(results)}
