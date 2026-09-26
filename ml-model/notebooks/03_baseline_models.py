"""
03_baseline_models.py
Baseline models for MDVR-KCL. Uses plain StratifiedKFold (not grouped) --
each row here is already one unique subject, so no leakage risk.
"""
import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

DATA_PATH = "data/mdvr_kcl_features.csv"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("subject_id", "status")]

X = df[feature_cols].values
y = df["status"].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def specificity_score(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

scoring = {
    "accuracy": "accuracy",
    "sensitivity_recall": make_scorer(recall_score, pos_label=1),
    "specificity": make_scorer(specificity_score),
    "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=1),
    "roc_auc": "roc_auc",
}

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ]),
}

results = {}
print(f"{'Model':<20}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'Prec':>8}{'F1':>8}{'AUC':>8}")
print("-" * 68)

for name, pipe in models.items():
    cv_results = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
    summary = {k.replace("test_", ""): (float(np.mean(v)), float(np.std(v)))
               for k, v in cv_results.items() if k.startswith("test_")}
    results[name] = summary
    print(f"{name:<20}"
          f"{summary['accuracy'][0]:>8.3f}"
          f"{summary['sensitivity_recall'][0]:>8.3f}"
          f"{summary['specificity'][0]:>8.3f}"
          f"{summary['precision'][0]:>8.3f}"
          f"{summary['f1'][0]:>8.3f}"
          f"{summary['roc_auc'][0]:>8.3f}")

with open(f"{OUT_DIR}/mdvr_baseline_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUT_DIR}/mdvr_baseline_results.json")