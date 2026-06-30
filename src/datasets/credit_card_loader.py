"""
Credit Card Fraud Detection Dataset (ULB)
Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Download: kaggle datasets download -d mlg-ulb/creditcardfraud
File: creditcard.csv

Columns: Time, V1-V28 (PCA), Amount, Class (0=normal, 1=fraud)
Size: 284,807 transactions | Fraud rate: 0.17%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_credit_card_fraud(csv_path: str, max_samples: int = 50000, random_state: int = 42):
    """
    Load and preprocess the ULB Credit Card Fraud dataset.

    Returns
    -------
    X : np.ndarray  shape (N, 30)  float32
    y : np.ndarray  shape (N,)     int64
    """
    df = pd.read_csv(csv_path)

    # Log-scale Amount (right-skewed); normalize Time
    df["Amount_log"] = np.log1p(df["Amount"])
    df["Time_norm"] = (df["Time"] - df["Time"].mean()) / (df["Time"].std() + 1e-9)

    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_log", "Time_norm"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["Class"].values.astype(np.int64)

    # Stratified subsample: keep all fraud, sample normals up to max_samples
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
        n_normal = max_samples - len(fraud_idx)
        sampled_normal = rng.choice(normal_idx, size=min(n_normal, len(normal_idx)), replace=False)
        idx = np.concatenate([fraud_idx, sampled_normal])
        rng.shuffle(idx)
        X, y = X[idx], y[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), y
