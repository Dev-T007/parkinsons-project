"""
09_batch_test.py
Tests every .wav file in a folder using the MDVR-KCL model and prints
a summary table. If you know some of them are healthy (no PD diagnosis),
list their filenames in KNOWN_HEALTHY to get a real accuracy number.
"""
import os
import sys
import joblib
import numpy as np
import librosa
sys.path.append("notebooks")
from importlib import import_module
extract_module = import_module("01_extract_dataset")

MODEL_DIR = "models"
TEST_DIR = "real_test_recordings"
KNOWN_HEALTHY = set({"dev.wav", "dhruvesh.wav", "param.wav"})

model = joblib.load(f"{MODEL_DIR}/mdvr_best_model.joblib")
feature_cols = joblib.load(f"{MODEL_DIR}/mdvr_feature_columns.joblib")

wav_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".wav")]
if not wav_files:
    print(f"No .wav files found in {TEST_DIR}/")
    sys.exit(1)

print(f"{'File':<25}{'Prediction':<15}{'Confidence':>12}")
print("-" * 52)

correct = 0
total_known = 0

for fname in sorted(wav_files):
    path = os.path.join(TEST_DIR, fname)
    try:
        y, sr = librosa.load(path, sr=None)
        y, _ = librosa.effects.trim(y, top_db=25)

        voice_feats = extract_module.extract_voice_features(path)
        mfcc_feats = extract_module.extract_mfcc_features(y, sr)
        speech_rate = extract_module.estimate_speech_rate(y, sr)

        feats = {}
        feats.update(voice_feats)
        feats.update(mfcc_feats)
        feats["speech_rate"] = speech_rate

        X = np.array([[feats[col] for col in feature_cols]])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        label = "Parkinson's" if pred == 1 else "Healthy"
        conf = prob[pred] * 100
        print(f"{fname:<25}{label:<15}{conf:>10.1f}%")

        if fname in KNOWN_HEALTHY:
            total_known += 1
            if pred == 0:
                correct += 1
    except Exception as e:
        print(f"{fname:<25}ERROR: {e}")

if total_known > 0:
    print(f"\nAccuracy on known-healthy test group: {correct}/{total_known} ({correct/total_known*100:.1f}%)")