import argparse
import importlib.util
import os
import subprocess
import tempfile

import joblib
import librosa
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

spec = importlib.util.spec_from_file_location("chunks", os.path.join(HERE, "10_extract_chunks.py"))
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)

model = joblib.load(os.path.join(BASE, "models", "mdvr_best_model_3.joblib"))
cols = joblib.load(os.path.join(BASE, "models", "mdvr_feature_columns_3.joblib"))
THR = float(joblib.load(os.path.join(BASE, "models", "mdvr_threshold_3.joblib")))


def chunk_table(wav_path):
    """Same chunking as training. Returns a DataFrame of chunk features."""
    y, sr = librosa.load(wav_path, sr=None)
    cl, hl = int(ex.CHUNK_SEC * sr), int(ex.HOP_SEC * sr)
    rows = []
    for start in range(0, max(1, len(y) - cl + 1), hl):
        try:
            f = ex.chunk_features(y[start:start + cl], sr)
        except Exception:
            continue
        if f["Fo_mean"] == 0 or any(pd.isna(v) for v in f.values()):
            continue
        rows.append(f)
    return pd.DataFrame(rows, columns=cols), sr, len(y) / sr


def person_prob(wav_path):
    t, _, _ = chunk_table(wav_path)
    return float(model.predict_proba(t[cols].values)[:, 1].mean()) if len(t) else np.nan


def ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


def roundtrip(folder, n):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".wav"))[:n]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for fn in files:
            src = os.path.join(folder, fn)
            rs, webm, dec = (os.path.join(tmp, x) for x in ("rs.wav", "a.webm", "dec.wav"))
            ff("-i", src, "-ac", "1", "-ar", "48000", rs)                                   # only resample
            ff("-i", src, "-ac", "1", "-ar", "48000", "-c:a", "libopus", "-b:a", "32k", webm)  # browser-like
            ff("-i", webm, dec)
            rows.append((fn[:14], person_prob(src), person_prob(rs), person_prob(dec)))
            print(f"  {rows[-1][0]:<15} direct {rows[-1][1]:.2f}   resampled-48k {rows[-1][2]:.2f}   "
                  f"opus-48k {rows[-1][3]:.2f}", flush=True)
    a = np.array([r[1:] for r in rows])
    print(f"\nMean P(PD):  direct {a[:,0].mean():.2f}   resampled-48k {a[:,1].mean():.2f}   opus-48k {a[:,2].mean():.2f}")
    for i, name in enumerate(["direct", "resampled-48k", "opus-48k"]):
        print(f"Flagged PD at {THR:.2f} ({name}): {(a[:, i] >= THR).sum()}/{len(a)}")
    print("\nBig jump at resampled-48k -> sample rate is the culprit."
          "\nBig jump only at opus-48k -> the browser codec is the culprit."
          "\nNo jump anywhere        -> your microphone / room is the problem, not the pipeline.")


def score_wav(path):
    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "in.wav")
        ff("-i", path, wav)                     # any format -> wav, native sample rate
        t, sr, dur = chunk_table(wav)
    if t.empty:
        print(f"{path}: no usable chunks"); return
    p = model.predict_proba(t[cols].values)[:, 1]
    print(f"\n=== {os.path.basename(path)}  ({dur:.0f}s, {sr} Hz, {len(t)} chunks)")
    print(f"mean P(PD) = {p.mean():.2f}   chunks: {np.round(p, 2).tolist()}")
    print(f"verdict at 0.50 -> {'PD' if p.mean() >= 0.5 else 'Healthy'};   "
          f"at tuned {THR:.2f} -> {'PD' if p.mean() >= THR else 'Healthy'}")
    ref = pd.read_csv(os.path.join(BASE, "data", "mdvr_kcl_chunks.csv"))
    hc = ref[ref.status == 0][cols]
    z = ((t.mean() - hc.mean()) / hc.std()).sort_values(key=np.abs, ascending=False)
    print("Features furthest from the HEALTHY training chunks (|z| > 3 = far outside):")
    for k in z.index[:7]:
        print(f"   {k:<14} z = {z[k]:+.1f}   yours {t[k].mean():.3f}   healthy avg {hc[k].mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--roundtrip", help="folder of healthy .wav files, e.g. data/mdvr-kcl/HC")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--wav", nargs="+", help="your own recording(s)")
    a = ap.parse_args()
    if a.roundtrip:
        roundtrip(a.roundtrip, a.n)
    for w in a.wav or []:
        score_wav(w)