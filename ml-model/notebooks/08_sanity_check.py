"""
08_sanity_check.py
Confirms the model isn't degenerate (i.e. doesn't just always predict one
class) by testing it on known Healthy and known Parkinson's rows directly
from the training data.
"""
import pandas as pd
import joblib
import numpy as np

MODEL_DIR = "models"
DATA_PATH = "data/mdvr_kcl_features.csv"

model = joblib.load(f"{MODEL_DIR}/mdvr_best_model.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/mdvr_feature_columns.joblib")

df = pd.read_csv(DATA_PATH)

healthy_rows = df[df["status"] == 0]
pd_rows = df[df["status"] == 1]

X_healthy = healthy_rows[feature_cols].values
X_pd = pd_rows[feature_cols].values

healthy_preds = model.predict(X_healthy)
pd_preds = model.predict(X_pd)

print(f"Known HEALTHY people ({len(X_healthy)} total):")
print(f"  Correctly predicted Healthy: {(healthy_preds == 0).sum()}/{len(X_healthy)}")
print(f"  Incorrectly predicted Parkinson's: {(healthy_preds == 1).sum()}/{len(X_healthy)}")

print(f"\nKnown PARKINSON'S people ({len(X_pd)} total):")
print(f"  Correctly predicted Parkinson's: {(pd_preds == 1).sum()}/{len(X_pd)}")
print(f"  Incorrectly predicted Healthy: {(pd_preds == 0).sum()}/{len(X_pd)}")