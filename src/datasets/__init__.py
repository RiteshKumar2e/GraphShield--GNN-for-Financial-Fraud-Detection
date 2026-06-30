from .credit_card_loader import load_credit_card_fraud
from .paysim_loader import load_paysim_fraud
from .insurance_loader import load_insurance_fraud
from .ecommerce_loader import load_ecommerce_fraud

DATASET_REGISTRY = {
    "credit_card": {
        "loader": load_credit_card_fraud,
        "domain": "Banking / Credit Card",
        "kaggle": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "file": "data/raw/credit_card/creditcard.csv",
        "description": "284,807 credit card transactions (ULB), 0.17% fraud",
    },
    "paysim": {
        "loader": load_paysim_fraud,
        "domain": "Mobile Payments / Fintech",
        "kaggle": "https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset",
        "file": "data/raw/paysim/online_payments_fraud.csv",
        "description": "6.3M+ simulated mobile payment transactions, ~0.13% fraud",
    },
    "insurance": {
        "loader": load_insurance_fraud,
        "domain": "Vehicle Insurance Claims",
        "kaggle": "https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection",
        "file": "data/raw/insurance/fraud_oracle.csv",
        "description": "15,420 vehicle insurance claims, 6% fraud",
    },
    "ecommerce": {
        "loader": load_ecommerce_fraud,
        "domain": "E-Commerce / Retail",
        "kaggle": "https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions",
        "file": "data/raw/ecommerce/Fraudulent_E-Commerce_Transaction_Data.csv",
        "description": "Online retail transactions with behavioral signals, ~10% fraud",
    },
}
