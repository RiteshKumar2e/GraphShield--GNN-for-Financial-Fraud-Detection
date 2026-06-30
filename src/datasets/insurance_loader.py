"""
Vehicle Insurance Claim Fraud Detection Dataset
Kaggle: https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection
Download: kaggle datasets download -d shivamb/vehicle-claim-fraud-detection
File: fraud_oracle.csv

Columns: Month, WeekOfMonth, DayOfWeek, Make, AccidentArea, DayOfWeekClaimed,
         MonthClaimed, WeekOfMonthClaimed, Sex, MaritalStatus, Age, Fault,
         PolicyType, VehicleCategory, VehiclePrice, FraudFound_P, PolicyNumber,
         RepNumber, Deductible, DriverRating, Days_Policy_Accident,
         Days_Policy_Claim, PastNumberOfClaims, AgeOfVehicle, AgeOfPolicyHolder,
         PoliceReportFiled, WitnessPresent, AgentType, NumberOfSuppliments,
         AddressChange_Claim, NumberOfCars, Year, BasePolicy
Size: 15,420 claims | Fraud rate: ~6%
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_insurance_fraud(csv_path: str, random_state: int = 42):
    """
    Load and preprocess the Vehicle Insurance Claim Fraud dataset.

    Returns
    -------
    X : np.ndarray  shape (N, F)  float32
    y : np.ndarray  shape (N,)    int64
    """
    df = pd.read_csv(csv_path)

    # Identify target column
    target_col = None
    for col in ["FraudFound_P", "fraud", "Fraud", "is_fraud", "isFraud"]:
        if col in df.columns:
            target_col = col
            break
    if target_col is None:
        raise ValueError(f"Target column not found. Available: {df.columns.tolist()}")

    y = df[target_col].values.astype(np.int64)
    df = df.drop(columns=[target_col])

    # Drop identifier columns
    id_cols = ["PolicyNumber", "RepNumber", "policy_id", "claim_id"]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])

    # Encode all categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = le.fit_transform(df[col].astype(str))

    # Impute missing values with median
    df = df.fillna(df.median(numeric_only=True))

    X = df.values.astype(np.float64)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), y
