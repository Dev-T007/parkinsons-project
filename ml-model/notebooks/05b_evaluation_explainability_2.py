"""
05b_explainability_2.py
Evaluation + explainability for the NEW chunk SVM model (leak-free).

Why not reuse 05_evaluation_explainability.py?
  - it uses plain StratifiedKFold (would leak on chunk data)
  - it uses SHAP TreeExplainer, which only works for tree models, not SVM

What this does:
  1. Same person-wise nested CV as 04_2 / 04b -> out-of-fold probability per person
  2. Applies the tuned threshold (models/mdvr_threshold_2.joblib)
  3. Person-level confusion matrix + classification report
  4. Permutation importance (works for ANY model, incl. SVM), measured on the
     held-out people of each fold -> which features the model really relies on

Saves NEW files only (into reports2/):
  mdvr_confusion_matrix_2.png, mdvr_classification_report_2.json,
  mdvr_feature_importance_2.png, mdvr_feature_importance_2.csv
Run from anywhere:  python notebooks/05b_explainability_2.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, roc_auc_score
from scipy.stats import uniform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml-model folder
DATA_PATH = os.path.join(BASE_DIR, "data", "mdvr_kcl_chunks.csv")
OUT_DIR = os.path.join(BASE_DIR, "reports2")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = joblib.load(os.path.join(MODEL_DIR, "mdvr_feature_columns_2.joblib"))
threshold = joblib.load(os.path.join(MODEL_DIR, "mdvr_threshold_2.joblib"))
X = df[feature_cols].values
y = df["status"].values
groups = df["subject_id"].values
print(f"Using threshold = {threshold}")

outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # same as 04_2 / 04b
inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)


def make_svm(X_tr, y_tr, g_tr):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])
    params = {"clf__C": uniform(0.1, 10), "clf__gamma": ["scale", 0.01, 0.03, 0.1]}
    search = RandomizedSearchCV(pipe, params, n_iter=15, scoring="roc_auc",
                                cv=inner_cv, n_jobs=1, random_state=42)
    search.fit(X_tr, y_tr, groups=g_tr)
    return search.best_estimator_


# ---------- 1. out-of-fold predictions + permutation importance per fold ----------
person_prob, person_true = {}, {}
fold_importances = []
for fold_no, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    model = make_svm(X[tr], y[tr], groups[tr])
    p = model.predict_proba(X[te])[:, 1]
    fold = pd.DataFrame({"sid": groups[te], "p": p, "y": y[te]})
    for sid, g in fold.groupby("sid"):
        person_prob[sid] = g["p"].mean()
        person_true[sid] = g["y"].iloc[0]

    # importance on UNSEEN people: shuffle one feature, see how much AUC drops
    pi = permutation_importance(model, X[te], y[te], scoring="roc_auc",
                                n_repeats=10, random_state=42, n_jobs=1)
    fold_importances.append(pi.importances_mean)
    print(f"Fold {fold_no} done")

sids = sorted(person_prob)
prob = np.array([person_prob[s] for s in sids])
true = np.array([person_true[s] for s in sids])
pred = (prob >= threshold).astype(int)

# ---------- 2. person-level evaluation ----------
print(f"\nPerson-level ROC-AUC: {roc_auc_score(true, prob):.3f}")
report = classification_report(true, pred, target_names=["Healthy", "Parkinson's"],
                               output_dict=True, zero_division=0)
print(classification_report(true, pred, target_names=["Healthy", "Parkinson's"], zero_division=0))
report["roc_auc"] = float(roc_auc_score(true, prob))
report["threshold"] = float(threshold)
with open(os.path.join(OUT_DIR, "mdvr_classification_report_2.json"), "w") as f:
    json.dump(report, f, indent=2)

cm = confusion_matrix(true, pred)
ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Parkinson's"]).plot(cmap="Blues", colorbar=False)
plt.title(f"Chunk SVM - person-level (threshold {threshold})")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mdvr_confusion_matrix_2.png"), dpi=150)
plt.close()
print("Confusion matrix (rows = true, cols = predicted):\n", cm)

# ---------- 3. feature importance ----------
imp = pd.DataFrame(fold_importances, columns=feature_cols)
summary = pd.DataFrame({"importance_mean": imp.mean(), "importance_std": imp.std()}) \
            .sort_values("importance_mean", ascending=False)
summary.to_csv(os.path.join(OUT_DIR, "mdvr_feature_importance_2.csv"))

top = summary.head(15)[::-1]
plt.figure(figsize=(8, 6))
plt.barh(top.index, top["importance_mean"], xerr=top["importance_std"], color="#4C72B0")
plt.xlabel("Drop in ROC-AUC when feature is shuffled")
plt.title("Permutation importance (chunk SVM, unseen people)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "mdvr_feature_importance_2.png"), dpi=150)
plt.close()

print("\nTop 10 features:")
print(summary.head(10).round(4).to_string())
print("\nSaved -> reports2/mdvr_confusion_matrix_2.png")
print("Saved -> reports2/mdvr_classification_report_2.json")
print("Saved -> reports2/mdvr_feature_importance_2.png / .csv")