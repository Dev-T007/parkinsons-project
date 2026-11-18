"""
04_advanced_models_2.py
Advanced models on the CHUNK dataset (data/mdvr_kcl_chunks.csv) -- leak-free.

Leakage protections:
  1. Person-wise split (StratifiedGroupKFold, groups = subject_id):
     all chunks of one person are in train OR test, never both.
  2. Nested CV: hyperparameter tuning happens INSIDE each outer training fold,
     also split person-wise, so tuning never sees the test people.
  3. Scaler inside the Pipeline -> fitted only on training chunks.
  4. Evaluation per PERSON: a person's chunk probabilities are averaged into
     one prediction, so scores are comparable with the old 37-row model.
  5. 'chunk' (chunk index) is NOT used as a feature -- same 25 features as the app.

Saves NEW files only (old model/reports untouched):
  models/mdvr_best_model_2.joblib
  models/mdvr_feature_columns_2.joblib
  reports2/mdvr_all_results_2.json
Run from anywhere:  python notebooks/04_advanced_models_2.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml-model folder
DATA_PATH = os.path.join(BASE_DIR, "data", "mdvr_kcl_chunks.csv")
OLD_RESULTS = os.path.join(BASE_DIR, "reports", "mdvr_all_results.json")
OUT_DIR = os.path.join(BASE_DIR, "reports2")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- data ----------------
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("subject_id", "status", "chunk")]
X = df[feature_cols].values
y = df["status"].values
groups = df["subject_id"].values
print(f"Chunks: {len(df)} | People: {df['subject_id'].nunique()} | Features: {len(feature_cols)}")

outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)


# ---------------- model builders (tuning inside each training fold) ----------------
def tune(pipe, params, X_tr, y_tr, g_tr, n_iter=15):
    search = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring="roc_auc",
                                cv=inner_cv, n_jobs=1, random_state=42)
    search.fit(X_tr, y_tr, groups=g_tr)
    return search.best_estimator_


def make_lr(X_tr, y_tr, g_tr):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))]
                    ).fit(X_tr, y_tr)


def make_svm(X_tr, y_tr, g_tr):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))])
    params = {"clf__C": uniform(0.1, 10), "clf__gamma": ["scale", 0.01, 0.03, 0.1]}
    return tune(pipe, params, X_tr, y_tr, g_tr)


def make_rf(X_tr, y_tr, g_tr):
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1))])
    params = {"clf__n_estimators": randint(150, 400), "clf__max_depth": randint(3, 12),
              "clf__min_samples_split": randint(2, 10), "clf__min_samples_leaf": randint(2, 10)}
    return tune(pipe, params, X_tr, y_tr, g_tr)


def make_xgb(X_tr, y_tr, g_tr):
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                           scale_pos_weight=spw, random_state=42, n_jobs=-1))])
    params = {"clf__n_estimators": randint(100, 400), "clf__max_depth": randint(2, 6),
              "clf__learning_rate": uniform(0.02, 0.18), "clf__subsample": uniform(0.7, 0.3),
              "clf__colsample_bytree": uniform(0.6, 0.4)}
    return tune(pipe, params, X_tr, y_tr, g_tr)


def make_ensemble(X_tr, y_tr, g_tr):
    ens = VotingClassifier(estimators=[("lr", make_lr(X_tr, y_tr, g_tr)),
                                       ("rf", make_rf(X_tr, y_tr, g_tr)),
                                       ("xgb", make_xgb(X_tr, y_tr, g_tr))], voting="soft")
    return ens.fit(X_tr, y_tr)   # refits copies of the already-tuned members


builders = {
    "LogisticRegression": make_lr,
    "SVM_RBF_tuned": make_svm,
    "RandomForest_tuned": make_rf,
    "XGBoost_tuned": make_xgb,
    "Ensemble_Voting": make_ensemble,
}


# ---------------- person-level metrics ----------------
def person_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    tn = np.sum((y_true == 0) & (pred == 0))
    fp = np.sum((y_true == 0) & (pred == 1))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "sensitivity_recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
    }


# ---------------- nested, person-wise evaluation ----------------
results = {}
print(f"\n{'Model':<22}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}   (person-level, 5-fold)")
for name, build in builders.items():
    person_prob, person_true = {}, {}
    for tr, te in outer_cv.split(X, y, groups):
        model = build(X[tr], y[tr], groups[tr])            # tuned on training people only
        p = model.predict_proba(X[te])[:, 1]
        fold = pd.DataFrame({"sid": groups[te], "p": p, "y": y[te]})
        for sid, g in fold.groupby("sid"):
            person_prob[sid] = g["p"].mean()               # average chunks -> 1 score per person
            person_true[sid] = g["y"].iloc[0]
    sids = sorted(person_prob)
    m = person_metrics(np.array([person_true[s] for s in sids]),
                       np.array([person_prob[s] for s in sids]))
    results[name] = m
    print(f"{name:<22}{m['accuracy']:>8.3f}{m['sensitivity_recall']:>8.3f}"
          f"{m['specificity']:>8.3f}{m['roc_auc']:>8.3f}")

best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
print(f"\nBest model by person-level ROC-AUC: {best_model_name} ({results[best_model_name]['roc_auc']:.3f})")

# ---------------- OLD vs NEW comparison ----------------
if os.path.exists(OLD_RESULTS):
    with open(OLD_RESULTS) as f:
        old = json.load(f)

    def _v(d, k):
        v = d.get(k)
        return v[0] if isinstance(v, list) else v

    old_best = max(old, key=lambda k: _v(old[k], "roc_auc"))
    o, n = old[old_best], results[best_model_name]
    print("\n===== OLD (whole recording) vs NEW (chunks) =====")
    print(f"{'':<38}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}")
    print(f"{'OLD  ' + old_best:<38}{_v(o, 'accuracy'):>8.3f}{_v(o, 'sensitivity_recall'):>8.3f}"
          f"{_v(o, 'specificity'):>8.3f}{_v(o, 'roc_auc'):>8.3f}")
    print(f"{'NEW  ' + best_model_name:<38}{n['accuracy']:>8.3f}{n['sensitivity_recall']:>8.3f}"
          f"{n['specificity']:>8.3f}{n['roc_auc']:>8.3f}")
    print("(note: OLD was tuned on all 37 people before CV, so it is slightly optimistic)")

# ---------------- final model on ALL data -> NEW files ----------------
final_model = builders[best_model_name](X, y, groups)
joblib.dump(final_model, os.path.join(MODEL_DIR, "mdvr_best_model_2.joblib"))
joblib.dump(feature_cols, os.path.join(MODEL_DIR, "mdvr_feature_columns_2.joblib"))
with open(os.path.join(OUT_DIR, "mdvr_all_results_2.json"), "w") as f:
    json.dump({"best_model": best_model_name, "person_level_results": results}, f, indent=2)

print(f"\nSaved -> models/mdvr_best_model_2.joblib")
print(f"Saved -> models/mdvr_feature_columns_2.joblib")
print(f"Saved -> reports2/mdvr_all_results_2.json")
print("Old mdvr_best_model.joblib is untouched.")