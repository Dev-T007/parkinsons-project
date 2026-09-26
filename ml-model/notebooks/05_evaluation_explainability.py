"""
05_evaluation_explainability.py
Confusion matrix + SHAP explainability for the MDVR-KCL best model.
"""
import pandas as pd
import numpy as np
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
import shap

DATA_PATH = "data/mdvr_kcl_features.csv"
OUT_DIR = "reports"
MODEL_DIR = "models"

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("subject_id", "status")]
X = df[feature_cols].values
y = df["status"].values

final_model = joblib.load(f"{MODEL_DIR}/mdvr_best_model.joblib")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_oof = cross_val_predict(final_model, X, y, cv=cv, n_jobs=1)

cm = confusion_matrix(y, y_pred_oof)
report = classification_report(y, y_pred_oof, target_names=["Healthy", "Parkinson's"], output_dict=True)
print(classification_report(y, y_pred_oof, target_names=["Healthy", "Parkinson's"]))

with open(f"{OUT_DIR}/mdvr_classification_report.json", "w") as f:
    json.dump(report, f, indent=2)

plt.figure(figsize=(5.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
            xticklabels=["Healthy", "Parkinson's"], yticklabels=["Healthy", "Parkinson's"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Out-of-Fold Confusion Matrix (MDVR-KCL)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mdvr_confusion_matrix.png", dpi=150)
plt.close()
print(f"Saved -> {OUT_DIR}/mdvr_confusion_matrix.png")

# SHAP
scaler = final_model.named_steps["scaler"]
clf = final_model.named_steps["clf"]
X_scaled = scaler.transform(X)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_scaled)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(sv, X_scaled, feature_names=feature_cols, show=False, plot_size=(10, 9))
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mdvr_shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_DIR}/mdvr_shap_summary.png")

mean_abs_shap = np.abs(sv).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
print("\nTop 10 features by mean |SHAP value|:")
print(shap_importance.head(10))