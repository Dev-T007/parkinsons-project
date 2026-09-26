"""
01_extract_dataset.py
Batch-extracts features from the MDVR-KCL dataset (HC + PD folders),
including MFCCs and speech rate — real connected-speech recordings.
"""
import os
import parselmouth
from parselmouth.praat import call
import librosa
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

DATA_DIR = "data/mdvr-kcl"
OUT_PATH = "data/mdvr_kcl_features.csv"

def extract_voice_features(wav_path):
    sound = parselmouth.Sound(wav_path)
    pitch = sound.to_pitch()
    f0_values = pitch.selected_array['frequency']
    f0_values = f0_values[f0_values != 0]
    fo_mean = np.mean(f0_values) if len(f0_values) > 0 else 0
    fo_max = np.max(f0_values) if len(f0_values) > 0 else 0
    fo_min = np.min(f0_values) if len(f0_values) > 0 else 0

    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)

    jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)

    shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    nhr = 10 ** (-hnr / 10) if hnr > 0 else 0

    return {
        "Fo_mean": fo_mean, "Fo_max": fo_max, "Fo_min": fo_min,
        "Jitter_local": jitter_local, "Jitter_rap": jitter_rap, "Jitter_ppq5": jitter_ppq5,
        "Shimmer_local": shimmer_local, "Shimmer_apq3": shimmer_apq3, "Shimmer_apq5": shimmer_apq5,
        "NHR": nhr, "HNR": hnr,
    }

def extract_mfcc_features(y, sr):
    y = librosa.util.normalize(y)  # normalize peak amplitude before MFCC extraction
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = mfccs.mean(axis=1)
    return {f"MFCC_{i+1}": float(mfcc_means[i]) for i in range(13)}

def estimate_speech_rate(y, sr):
    hop_length = 512
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_smooth = uniform_filter1d(rms, size=5)
    threshold = np.mean(rms_smooth) * 0.6
    min_distance = int(sr / hop_length * 0.15)
    peaks, _ = find_peaks(rms_smooth, height=threshold, distance=max(min_distance, 1))
    duration_sec = len(y) / sr
    return len(peaks) / duration_sec if duration_sec > 0 else 0

def process_file(wav_path, label, subject_id):
    y, sr = librosa.load(wav_path, sr=None)
    y, _ = librosa.effects.trim(y, top_db=25)

    voice_feats = extract_voice_features(wav_path)
    mfcc_feats = extract_mfcc_features(y, sr)
    speech_rate = estimate_speech_rate(y, sr)

    row = {"subject_id": subject_id, "status": label}
    row.update(voice_feats)
    row.update(mfcc_feats)
    row["speech_rate"] = speech_rate
    return row

if __name__ == "__main__":
    rows = []
    for label, folder in [(0, "HC"), (1, "PD")]:
        folder_path = os.path.join(DATA_DIR, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        print(f"Processing {len(files)} files in {folder}/ ...")
        for fname in files:
            wav_path = os.path.join(folder_path, fname)
            subject_id = os.path.splitext(fname)[0]
            try:
                row = process_file(wav_path, label, subject_id)
                rows.append(row)
                print(f"  OK: {fname}")
            except Exception as e:
                print(f"  FAILED: {fname} -> {e}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} rows -> {OUT_PATH}")
    print(f"\nClass balance:\n{df['status'].value_counts()}")