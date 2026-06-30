from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.multi_graph_service import get_multi_graph_service

router = APIRouter(tags=["Multi-Dataset Graphs"])


class SubgraphRequest(BaseModel):
    node_id:   int
    k:         int = 2
    max_nodes: int = 40


@router.get("/datasets", summary="List available trained dataset graphs")
def list_datasets():
    svc   = get_multi_graph_service()
    keys  = svc.available_keys()
    return [svc.get_stats(k) for k in keys]


@router.get("/datasets/{key}/sample", summary="Sample high-fraud subgraph for initial display")
def sample_subgraph(key: str):
    svc = get_multi_graph_service()
    if key not in svc.available_keys():
        raise HTTPException(status_code=404, detail=f"Dataset '{key}' not loaded. Run Notebook 10 first.")
    node_id      = svc.get_sample_node(key)
    nodes, edges = svc.k_hop_subgraph(key, node_id, k=2, max_nodes=40)
    stats        = svc.get_stats(key)
    return {
        'dataset_key':  key,
        'dataset_name': stats['name'],
        'target_node':  node_id,
        'nodes':        nodes,
        'edges':        edges,
        'stats':        stats,
    }


@router.post("/datasets/{key}/subgraph", summary="K-hop subgraph for any node")
def get_subgraph(key: str, req: SubgraphRequest):
    svc = get_multi_graph_service()
    if key not in svc.available_keys():
        raise HTTPException(status_code=404, detail=f"Dataset '{key}' not loaded. Run Notebook 10 first.")
    stats = svc.get_stats(key)
    if req.node_id < 0 or req.node_id >= stats['num_nodes']:
        raise HTTPException(status_code=400,
                            detail=f"node_id {req.node_id} out of range [0, {stats['num_nodes']-1}]")
    k         = max(1, min(req.k, 3))
    max_nodes = max(10, min(req.max_nodes, 80))
    nodes, edges = svc.k_hop_subgraph(key, req.node_id, k=k, max_nodes=max_nodes)
    return {
        'dataset_key': key,
        'target_node': req.node_id,
        'nodes':       nodes,
        'edges':       edges,
        'stats':       stats,
    }
