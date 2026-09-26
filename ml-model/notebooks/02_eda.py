"""
02_eda.py
EDA for the MDVR-KCL feature dataset — includes MFCCs and speech rate
this time, since it's real connected speech.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/mdvr_kcl_features.csv"
OUT_DIR = "reports"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("\nMissing values:", df.isnull().sum().sum())

feature_cols = [c for c in df.columns if c not in ("subject_id", "status")]
print(f"\nNumber of features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

print("\n--- Class balance ---")
print(df["status"].value_counts())
print((df["status"].value_counts(normalize=True) * 100).round(2), "%")

# Correlation heatmap
plt.figure(figsize=(16, 14))
corr = df[feature_cols + ["status"]].corr()
sns.heatmap(corr, cmap="RdBu_r", center=0, square=True)
plt.title("Feature Correlation Heatmap (MDVR-KCL)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mdvr_correlation_heatmap.png", dpi=150)
plt.close()

target_corr = corr["status"].drop("status").abs().sort_values(ascending=False)
print("\n--- Top 15 features most correlated with status ---")
print(target_corr.head(15))

# Distribution plots for top features
top_features = target_corr.head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, feat in zip(axes.flatten(), top_features):
    sns.kdeplot(data=df, x=feat, hue=df["status"].map({0: "Healthy", 1: "Parkinson's"}),
                fill=True, ax=ax, palette=["#A78BFA", "#6D28D9"], common_norm=False)
    ax.set_title(feat)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/mdvr_top_feature_distributions.png", dpi=150)
plt.close()

print("\nEDA complete.")