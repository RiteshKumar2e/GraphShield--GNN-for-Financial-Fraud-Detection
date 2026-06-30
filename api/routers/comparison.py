from fastapi import APIRouter
from pathlib import Path
import pandas as pd

router = APIRouter(tags=["Comparison"])

_CSV      = Path("results/multi_dataset/cross_dataset_comparison.csv")
_BEST_CSV = Path("results/multi_dataset/best_model_per_dataset.csv")

_DATASET_META = {
    "Elliptic Bitcoin": {
        "domain": "Blockchain / Crypto",
        "nodes": "46,564",
        "fraud_rate": "9.77%",
        "graph_type": "Natural BTC flow graph",
        "kaggle": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set",
    },
    "Credit Card": {
        "domain": "Banking / Credit Card",
        "nodes": "284,807",
        "fraud_rate": "0.17%",
        "graph_type": "KNN(k=10) on PCA features",
        "kaggle": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    },
    "PaySim Payments": {
        "domain": "Mobile Payments / Fintech",
        "nodes": "6.3M+",
        "fraud_rate": "0.13%",
        "graph_type": "KNN(k=10) on tx features",
        "kaggle": "https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset",
    },
    "Vehicle Insurance": {
        "domain": "Vehicle Insurance Claims",
        "nodes": "15,420",
        "fraud_rate": "6.0%",
        "graph_type": "KNN(k=10) on mixed features",
        "kaggle": "https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection",
    },
    "E-Commerce": {
        "domain": "Online Retail",
        "nodes": "1.47M",
        "fraud_rate": "~10%",
        "graph_type": "KNN(k=10) on behavioural",
        "kaggle": "https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions",
    },
}


@router.get("/comparison", summary="Multi-dataset GNN comparison results")
def get_comparison():
    results, best = [], []

    if _CSV.exists():
        df = pd.read_csv(_CSV)
        results = df.to_dict(orient="records")

    if _BEST_CSV.exists():
        df_best = pd.read_csv(_BEST_CSV)
        best = df_best.to_dict(orient="records")

    return {
        "available":        _CSV.exists(),
        "results":          results,
        "best_per_dataset": best,
        "dataset_meta":     _DATASET_META,
    }
