"""
Fraudulent E-Commerce Transactions Dataset
Kaggle: https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions
Download: kaggle datasets download -d shriyashjagtap/fraudulent-e-commerce-transactions
File: Fraudulent_E-Commerce_Transaction_Data.csv

Columns (typical): Transaction ID, Customer ID, Transaction Amount, Transaction Date,
                   Payment Method, Product Category, Quantity, Customer Age,
                   Customer Location, Device Used, IP Address, Shipping Address,
                   Billing Address, Is Fraudulent, Account Age Days, Transaction Hour
Size: ~1.47M rows | Fraud rate: ~10%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_ecommerce_fraud(csv_path: str, max_samples: int = 50000, random_state: int = 42):
    """
    Load and preprocess the E-Commerce Fraud dataset.

    Returns
    -------
    X : np.ndarray  shape (N, F)  float32
    y : np.ndarray  shape (N,)    int64
    """
    df = pd.read_csv(csv_path)

    # Identify target column (flexible naming)
    target_col = None
    for col in ["Is Fraudulent", "is_fraudulent", "isFraud", "fraud", "Fraud",
                "is_fraud", "label", "Label", "Class", "class"]:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError(f"Target column not found. Available: {df.columns.tolist()}")

    y = df[target_col].values.astype(np.int64)
    df = df.drop(columns=[target_col])

    # Drop high-cardinality / ID / datetime columns
    drop_keywords = ["id", "date", "ip address", "address", "location"]
    drop_cols = [c for c in df.columns if any(k in c.lower() for k in drop_keywords)]
    df = df.drop(columns=drop_cols, errors="ignore")

    # Encode remaining categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    df = df.fillna(df.median(numeric_only=True))

    X = df.values.astype(np.float64)

    # Stratified subsample for large datasets
    if len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        fraud_idx = np.where(y == 1)[0]
        normal_idx = np.where(y == 0)[0]
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
