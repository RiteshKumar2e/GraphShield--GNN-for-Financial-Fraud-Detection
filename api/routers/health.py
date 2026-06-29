import time
import datetime
from fastapi import APIRouter
from api.schemas import HealthResponse, StatsResponse
from api.services.model_service import get_model_service
from api.services.graph_service import get_graph_service

router = APIRouter(tags=["System"])
_start_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="Service health check")
def health():
    ms = get_model_service()
    gs = get_graph_service()
    return HealthResponse(
        status         = "ok" if (ms.loaded and gs.loaded) else "loading",
        model_loaded   = ms.loaded,
        graph_loaded   = gs.loaded,
        total_nodes    = gs.num_nodes if gs.loaded else 0,
        total_edges    = gs.num_edges if gs.loaded else 0,
        uptime_seconds = round(time.time() - _start_time, 1),
        timestamp      = datetime.datetime.utcnow().isoformat() + "Z",
    )


@router.get("/stats", response_model=StatsResponse, summary="Dataset and model statistics")
def stats():
    ms = get_model_service()
    gs = get_graph_service()
    return StatsResponse(
        total_nodes         = gs.num_nodes if gs.loaded else 0,
        total_edges         = gs.num_edges if gs.loaded else 0,
        illicit_nodes       = gs.num_illicit if gs.loaded else 0,
        licit_nodes         = gs.num_licit if gs.loaded else 0,
        fraud_rate          = round(gs.num_illicit / gs.num_nodes * 100, 2) if gs.loaded else 0.0,
        avg_degree          = gs.avg_degree if gs.loaded else 0.0,
        predictions_served  = ms.predictions_served,
        model_name          = ms.model_name,
    )
