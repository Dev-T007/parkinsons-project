"""
06_predict.py
Predicts on a real recording using the MDVR-KCL trained model.
Expects a recording of someone READING TEXT ALOUD (not a sustained vowel) --
this model was trained on connected speech, not phonation.
"""
import sys
import joblib
import numpy as np
import librosa
sys.path.append("notebooks")
from importlib import import_module
extract_module = import_module("01_extract_dataset")

MODEL_DIR = "models"

def predict(wav_path):
    model = joblib.load(f"{MODEL_DIR}/mdvr_best_model.joblib")
    feature_cols = joblib.load(f"{MODEL_DIR}/mdvr_feature_columns.joblib")

    y, sr = librosa.load(wav_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=25)

    voice_feats = extract_module.extract_voice_features(wav_path)
    mfcc_feats = extract_module.extract_mfcc_features(y, sr)
    speech_rate = extract_module.estimate_speech_rate(y, sr)

    feats = {}
    feats.update(voice_feats)
    feats.update(mfcc_feats)
    feats["speech_rate"] = speech_rate

    X = np.array([[feats[col] for col in feature_cols]])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    label = "Parkinson's" if prediction == 1 else "Healthy"
    confidence = probability[prediction] * 100

    print(f"\n{'='*40}")
    print(f"File: {wav_path}")
    print(f"{'='*40}")
    print(f"Result: {label}")
    print(f"Model confidence: {confidence:.1f}%")
    print(f"\nFull breakdown:")
    print(f"  Healthy:     {probability[0]*100:.1f}%")
    print(f"  Parkinson's: {probability[1]*100:.1f}%")
    print(f"{'='*40}")

if __name__ == "__main__":
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "test_reading.wav"
    predict(wav_path)