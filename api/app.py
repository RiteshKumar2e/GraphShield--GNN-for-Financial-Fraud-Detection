"""
GraphShield API — Explainable GNN Fraud Detection Service
Run:  uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from api.services.model_service import get_model_service
from api.services.graph_service import get_graph_service
from api.services.multi_graph_service import get_multi_graph_service
from api.routers import score, explain, health, comparison
from api.routers import multi_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    print("GraphShield API starting…")
    gs = get_graph_service()
    gs.load()
    print(f"  Graph loaded: {gs.num_nodes:,} nodes, {gs.num_edges:,} edges")

    ms = get_model_service()
    ms.load()
    print(f"  Model loaded: {ms.model_name}")

    mgs = get_multi_graph_service()
    mgs.load()

    yield  # app runs here

    # shutdown
    print("GraphShield API shutting down.")


app = FastAPI(
    title       = "GraphShield API",
    description = "Explainable Graph Neural Network for Financial Fraud Detection",
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/api/docs",
    redoc_url   = "/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Routers
app.include_router(score.router,       prefix="/api/v1")
app.include_router(explain.router,     prefix="/api/v1")
app.include_router(health.router,      prefix="/api/v1")
app.include_router(comparison.router,  prefix="/api/v1")
app.include_router(multi_graph.router, prefix="/api/v1")

# Serve dashboard
_dashboard = Path("dashboard")
if _dashboard.exists():
    app.mount("/static", StaticFiles(directory=str(_dashboard)), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        return FileResponse(str(_dashboard / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)
