"""
07_diagnose_features.py
Compares extracted features from a real recording against the MDVR-KCL
training data ranges, to see which specific features are out of range.
"""
import sys
import pandas as pd
import numpy as np
import librosa
sys.path.append("notebooks")
from importlib import import_module
extract_module = import_module("01_extract_dataset")

DATA_PATH = "data/mdvr_kcl_features.csv"

df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c not in ("subject_id", "status")]
stats = df[feature_cols].agg(["mean", "std", "min", "max"]).T

wav_path = sys.argv[1]
y, sr = librosa.load(wav_path, sr=None)
y, _ = librosa.effects.trim(y, top_db=25)

voice_feats = extract_module.extract_voice_features(wav_path)
mfcc_feats = extract_module.extract_mfcc_features(y, sr)
speech_rate = extract_module.estimate_speech_rate(y, sr)

feats = {}
feats.update(voice_feats)
feats.update(mfcc_feats)
feats["speech_rate"] = speech_rate

duration_sec = len(y) / sr
print(f"\nRecording duration: {duration_sec:.1f} seconds")
print(f"(Training data recordings are roughly 20-40 seconds for the full passage)\n")

print(f"Comparing {wav_path} against training data ranges:\n")
print(f"{'Feature':<18}{'Your Value':>14}{'Dataset Mean':>14}{'Dataset Range':>22}{'Z-score':>10}")
print("-" * 82)
for feat in feature_cols:
    val = feats[feat]
    mean = stats.loc[feat, "mean"]
    std = stats.loc[feat, "std"]
    lo, hi = stats.loc[feat, "min"], stats.loc[feat, "max"]
    z = (val - mean) / std if std > 0 else 0
    flag = "  <-- OUT OF RANGE" if (val < lo or val > hi) else ""
    print(f"{feat:<18}{val:>14.5f}{mean:>14.5f}{f'[{lo:.4f}, {hi:.4f}]':>22}{z:>10.2f}{flag}")