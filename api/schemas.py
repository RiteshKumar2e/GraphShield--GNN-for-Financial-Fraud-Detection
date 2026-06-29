from pydantic import BaseModel, Field
from typing import Optional
import datetime


class ScoreRequest(BaseModel):
    node_id: int = Field(..., description="Transaction node index in the graph", example=7929)

    model_config = {"json_schema_extra": {"examples": [{"node_id": 7929}]}}


class EdgeInfo(BaseModel):
    source: int
    target: int
    weight: float


class ScoreResponse(BaseModel):
    node_id: int
    fraud_probability: float
    licit_probability: float
    verdict: str                          # "ILLICIT" | "LICIT" | "REVIEW"
    confidence: str                        # "HIGH" | "MEDIUM" | "LOW"
    top_features: list[dict]
    subgraph_nodes: list[int]
    subgraph_edges: list[EdgeInfo]
    threshold_used: float
    timestamp: str


class ExplainRequest(BaseModel):
    node_id: int
    top_k_edges: int = Field(default=20, ge=5, le=50)


class ExplainResponse(BaseModel):
    node_id: int
    fraud_probability: float
    explanation_edges: list[EdgeInfo]
    explanation_nodes: list[dict]          # [{id, label, fraud_prob}]
    top_features: list[dict]              # [{index, name, importance}]
    summary: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    graph_loaded: bool
    total_nodes: int
    total_edges: int
    uptime_seconds: float
    timestamp: str


class StatsResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    total_nodes: int
    total_edges: int
    illicit_nodes: int
    licit_nodes: int
    fraud_rate: float
    avg_degree: float
    predictions_served: int
    model_name: str
