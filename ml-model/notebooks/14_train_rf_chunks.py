import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ml-model folder
df = pd.read_csv(os.path.join(BASE, "data", "mdvr_kcl_chunks.csv"))
DROP = ["MFCC_1", "MFCC_2", "MFCC_8", "MFCC_11"]   # microphone-sensitive features (leave [] to use all 25)
cols = [c for c in df.columns if c not in ("subject_id", "status", "chunk") + tuple(DROP)]
print(f"{len(df)} chunks | {df.subject_id.nunique()} people | {len(cols)} features")


def make():
    # shallow trees + balanced classes: cautious settings for only 37 people
    return RandomForestClassifier(n_estimators=250, max_depth=6, min_samples_leaf=5,
                                  class_weight="balanced", random_state=42, n_jobs=-1)


# person-wise test (all chunks of a person stay together), repeated with 6 different splits
aucs, sens, spec = [], [], []
for seed in range(6):
    oof = np.zeros(len(df))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df, df.status, df.subject_id):
        oof[te] = make().fit(df.iloc[tr][cols].values, df.status.iloc[tr]).predict_proba(df.iloc[te][cols].values)[:, 1]
    p = (pd.DataFrame({"g": df.subject_id, "y": df.status, "p": oof})
           .groupby("g").agg(y=("y", "first"), p=("p", "mean")))      # average chunks -> 1 score per person
    pred = p.p >= 0.5
    aucs.append(roc_auc_score(p.y, p.p))
    sens.append(pred[p.y == 1].mean())
    spec.append((~pred)[p.y == 0].mean())
print("\nPerson-level result at plain 0.5 (6 repeated splits):")
print(f"  AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   sensitivity {np.mean(sens):.2f}   specificity {np.mean(spec):.2f}")

final = make().fit(df[cols].values, df.status.values)
joblib.dump(final, os.path.join(BASE, "models", "mdvr_best_model_3.joblib"))
joblib.dump(cols, os.path.join(BASE, "models", "mdvr_feature_columns_3.joblib"))
joblib.dump(0.5, os.path.join(BASE, "models", "mdvr_threshold_3.joblib"))
print("\nSaved models/mdvr_best_model_3.joblib, mdvr_feature_columns_3.joblib, mdvr_threshold_3.joblib")