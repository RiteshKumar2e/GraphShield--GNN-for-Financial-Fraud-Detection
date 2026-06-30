"""
PaySim Online Payments Fraud Detection Dataset
Kaggle: https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset
Download: kaggle datasets download -d rupakroy/online-payments-fraud-detection-dataset
File: online_payments_fraud.csv  (or PS_20174392719_1491204439457_log.csv for original PaySim)

Columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
         nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
Size: 6.3M+ transactions | Fraud rate: ~0.13%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_paysim_fraud(csv_path: str, max_samples: int = 50000, random_state: int = 42):
    """
    Load and preprocess the PaySim mobile payments fraud dataset.

    Returns
    -------
    X : np.ndarray  shape (N, F)  float32
    y : np.ndarray  shape (N,)    int64
    """
    df = pd.read_csv(csv_path)

    # Drop non-feature columns
    drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Encode transaction type
    if "type" in df.columns:
        le = LabelEncoder()
        df["type"] = le.fit_transform(df["type"].astype(str))

    # Target
    target_col = "isFraud" if "isFraud" in df.columns else "is_fraud"
    y = df[target_col].values.astype(np.int64)
    df = df.drop(columns=[target_col])

    # Log-scale amount
    if "amount" in df.columns:
        df["amount"] = np.log1p(df["amount"])

    X = df.values.astype(np.float64)

    # Stratified subsample (dataset has millions of rows)
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        # Balance: keep up to 20% fraud in sample so the graph has meaningful signal
        n_fraud = min(len(fraud_idx), max_samples // 5)
        n_normal = max_samples - n_fraud
        s_fraud = rng.choice(fraud_idx, size=n_fraud, replace=False)
        s_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
        idx = np.concatenate([s_fraud, s_normal])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), y
