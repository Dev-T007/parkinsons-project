"""
04_advanced_models.py
Random Forest + XGBoost for MDVR-KCL. Search space kept small given
only 37 samples -- aggressive tuning on this little data risks overfitting
the tuning process itself, not just the model.
"""
import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

DATA_PATH = "data/mdvr_kcl_features.csv"
OUT_DIR = "reports"
MODEL_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("subject_id", "status")]
X = df[feature_cols].values
y = df["status"].values

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)

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

# Random Forest -- small search space
rf_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
])
rf_param_dist = {
    "clf__n_estimators": randint(100, 300),
    "clf__max_depth": randint(2, 8),
    "clf__min_samples_split": randint(2, 6),
    "clf__min_samples_leaf": randint(1, 4),
}
print("Tuning Random Forest ...")
rf_search = RandomizedSearchCV(
    rf_pipe, rf_param_dist, n_iter=20, scoring="roc_auc",
    cv=list(inner_cv.split(X, y)), n_jobs=1, random_state=42,
)
rf_search.fit(X, y)
print("Best RF params:", rf_search.best_params_)
best_rf = rf_search.best_estimator_

# XGBoost -- small search space
scale_pos_weight = (y == 0).sum() / (y == 1).sum()
xgb_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                           scale_pos_weight=scale_pos_weight, random_state=42)),
])
xgb_param_dist = {
    "clf__n_estimators": randint(50, 200),
    "clf__max_depth": randint(2, 5),
    "clf__learning_rate": uniform(0.02, 0.18),
    "clf__subsample": uniform(0.7, 0.3),
}
print("Tuning XGBoost ...")
xgb_search = RandomizedSearchCV(
    xgb_pipe, xgb_param_dist, n_iter=20, scoring="roc_auc",
    cv=list(inner_cv.split(X, y)), n_jobs=1, random_state=42,
)
xgb_search.fit(X, y)
print("Best XGB params:", xgb_search.best_params_)
best_xgb = xgb_search.best_estimator_

results = {}
print(f"\n{'Model':<22}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}")
for name, model in [("RandomForest_tuned", best_rf), ("XGBoost_tuned", best_xgb)]:
    cv_results = cross_validate(model, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
    summary = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_results.items() if k.startswith("test_")}
    results[name] = summary
    print(f"{name:<22}{summary['accuracy']:>8.3f}{summary['sensitivity_recall']:>8.3f}"
          f"{summary['specificity']:>8.3f}{summary['roc_auc']:>8.3f}")

# Ensemble
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
])
ensemble = VotingClassifier(estimators=[("lr", lr_pipe), ("rf", best_rf), ("xgb", best_xgb)], voting="soft")
cv_results = cross_validate(ensemble, X, y, cv=outer_cv, scoring=scoring, n_jobs=1)
summary = {k.replace("test_", ""): float(np.mean(v)) for k, v in cv_results.items() if k.startswith("test_")}
results["Ensemble_Voting"] = summary
print(f"{'Ensemble_Voting':<22}{summary['accuracy']:>8.3f}{summary['sensitivity_recall']:>8.3f}"
      f"{summary['specificity']:>8.3f}{summary['roc_auc']:>8.3f}")

with open(f"{OUT_DIR}/mdvr_baseline_results.json") as f:
    baseline_results = json.load(f)
all_results = {**baseline_results, **results}
with open(f"{OUT_DIR}/mdvr_all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

auc_scores = {k: v["roc_auc"] if "roc_auc" in v and isinstance(v["roc_auc"], float) else v["roc_auc"][0] for k, v in all_results.items()}
best_model_name = max(auc_scores, key=auc_scores.get)
print(f"\nBest model by mean ROC-AUC: {best_model_name} ({auc_scores[best_model_name]:.3f})")

name_to_model = {"RandomForest_tuned": best_rf, "XGBoost_tuned": best_xgb, "Ensemble_Voting": ensemble,
                  "LogisticRegression": lr_pipe,
                  "SVM_RBF": Pipeline([("scaler", StandardScaler()),
                                       ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])}
final_model = name_to_model[best_model_name]
final_model.fit(X, y)
joblib.dump(final_model, f"{MODEL_DIR}/mdvr_best_model.joblib")
joblib.dump(feature_cols, f"{MODEL_DIR}/mdvr_feature_columns.joblib")
print(f"Saved -> {MODEL_DIR}/mdvr_best_model.joblib")