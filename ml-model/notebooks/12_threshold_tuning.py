"""
04b_threshold_tuning.py
Finds a better decision threshold for the chunk SVM model (mdvr_best_model_2).

Default threshold 0.5 gave high specificity but low sensitivity (misses PD patients).
Here we:
  1. Re-run the SAME leak-free nested, person-wise CV for SVM_RBF_tuned
     and collect each person's out-of-fold probability (avg of their chunks).
  2. Try thresholds 0.20 -> 0.60 and pick the one with the best balance
     (Youden's J = sensitivity + specificity - 1), preferring higher sensitivity on ties.
  3. Save the threshold -> models/mdvr_threshold_2.joblib (the app will use it in Step 3).

Run from anywhere:  python notebooks/04b_threshold_tuning.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from scipy.stats import uniform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml-model folder
DATA_PATH = os.path.join(BASE_DIR, "data", "mdvr_kcl_chunks.csv")
OUT_DIR = os.path.join(BASE_DIR, "reports2")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
feature_cols = joblib.load(os.path.join(MODEL_DIR, "mdvr_feature_columns_2.joblib"))
X = df[feature_cols].values
y = df["status"].values
groups = df["subject_id"].values

outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # same as 04_2
inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)


def make_svm(X_tr, y_tr, g_tr):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])
    params = {"clf__C": uniform(0.1, 10), "clf__gamma": ["scale", 0.01, 0.03, 0.1]}
    search = RandomizedSearchCV(pipe, params, n_iter=15, scoring="roc_auc",
                                cv=inner_cv, n_jobs=1, random_state=42)
    search.fit(X_tr, y_tr, groups=g_tr)
    return search.best_estimator_


# ---------- 1. out-of-fold person probabilities ----------
person_prob, person_true = {}, {}
for fold_no, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
    model = make_svm(X[tr], y[tr], groups[tr])
    p = model.predict_proba(X[te])[:, 1]
    fold = pd.DataFrame({"sid": groups[te], "p": p, "y": y[te]})
    for sid, g in fold.groupby("sid"):
        person_prob[sid] = g["p"].mean()
        person_true[sid] = g["y"].iloc[0]
    print(f"Fold {fold_no} done")

sids = sorted(person_prob)
prob = np.array([person_prob[s] for s in sids])
true = np.array([person_true[s] for s in sids])
print(f"\nPerson-level ROC-AUC: {roc_auc_score(true, prob):.3f}  (threshold does not change AUC)")


# ---------- 2. try thresholds ----------
def scores_at(t):
    pred = (prob >= t).astype(int)
    tp = np.sum((true == 1) & (pred == 1)); fn = np.sum((true == 1) & (pred == 0))
    tn = np.sum((true == 0) & (pred == 0)); fp = np.sum((true == 0) & (pred == 1))
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / len(true)
    return {"threshold": round(float(t), 2), "accuracy": float(acc),
            "sensitivity": float(sens), "specificity": float(spec), "youden_j": float(sens + spec - 1)}

table = [scores_at(t) for t in np.arange(0.20, 0.61, 0.01)]
best = max(table, key=lambda r: (r["youden_j"], r["sensitivity"]))

print(f"\n{'Thresh':>7}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'J':>8}")
for r in table:
    if round(r["threshold"] * 100) % 5 == 0 or r is best:
        mark = "  <-- chosen" if r is best else ("  (default)" if abs(r["threshold"] - 0.5) < 1e-9 else "")
        print(f"{r['threshold']:>7.2f}{r['accuracy']:>8.3f}{r['sensitivity']:>8.3f}"
              f"{r['specificity']:>8.3f}{r['youden_j']:>8.3f}{mark}")

# ---------- 3. save ----------
joblib.dump(best["threshold"], os.path.join(MODEL_DIR, "mdvr_threshold_2.joblib"))
with open(os.path.join(OUT_DIR, "mdvr_threshold_results_2.json"), "w") as f:
    json.dump({"chosen": best, "default_0.5": scores_at(0.5), "all": table}, f, indent=2)

print(f"\nChosen threshold: {best['threshold']:.2f}")
print("Saved -> models/mdvr_threshold_2.joblib")
print("Saved -> reports2/mdvr_threshold_results_2.json")
print("Note: threshold picked on the same CV predictions, so these numbers are slightly optimistic.")