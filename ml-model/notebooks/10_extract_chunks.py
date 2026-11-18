"""
10_extract_chunks.py
Creates a NEW chunk-level dataset from MDVR-KCL (does not touch the old one).
Each recording is cut into 8-second overlapping chunks, and the SAME 25
features are extracted per chunk using process_file() from 01_extract_dataset.py.

Output: data/mdvr_kcl_chunks.csv  (subject_id, status, chunk, + 25 features)
Run from the ml-model folder:  python notebooks/10_extract_chunks.py
"""
import os
import tempfile
import importlib.util
import pandas as pd
import librosa
import soundfile as sf

# --- load 01_extract_dataset.py (name starts with a number, so use importlib) ---
_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_extract_dataset.py")
_spec = importlib.util.spec_from_file_location("extract01", _path)
extract01 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract01)   # main loop of 01 does NOT run (it's under __main__)

DATA_DIR = "data/mdvr-kcl"              # same input as 01_extract_dataset.py
OUT_PATH = "data/mdvr_kcl_chunks.csv"   # NEW file -> old CSV stays untouched
CHUNK_SEC = 8.0   # same as the app's minimum recording length
HOP_SEC = 4.0     # 50% overlap between chunks


def chunk_features(y, sr):
    """Save one chunk as a temp wav, run 01's process_file() on it, return the 25 features."""
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp_path, y, sr)
        row = extract01.process_file(tmp_path, label=0, subject_id="tmp")
        row.pop("subject_id")
        row.pop("status")
        return row
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    rows = []
    for label, folder in [(0, "HC"), (1, "PD")]:
        folder_path = os.path.join(DATA_DIR, folder)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]
        print(f"Processing {len(files)} files in {folder}/ ...")
        for fname in files:
            subject_id = os.path.splitext(fname)[0]
            y, sr = librosa.load(os.path.join(folder_path, fname), sr=None)
            chunk_len, hop_len = int(CHUNK_SEC * sr), int(HOP_SEC * sr)

            kept = 0
            for idx, start in enumerate(range(0, max(1, len(y) - chunk_len + 1), hop_len)):
                seg = y[start:start + chunk_len]
                try:
                    feats = chunk_features(seg, sr)
                except Exception:
                    continue                      # Praat fails on silent chunks -> skip
                if feats["Fo_mean"] == 0 or any(pd.isna(v) for v in feats.values()):
                    continue                      # no usable voice in this chunk -> skip
                row = {"subject_id": subject_id, "status": label, "chunk": idx}
                row.update(feats)
                rows.append(row)
                kept += 1
            print(f"  OK: {fname} -> {kept} chunks")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    per_person = df.groupby("subject_id").size()
    print(f"\nSaved {len(df)} chunk rows -> {OUT_PATH}")
    print(f"People: {df['subject_id'].nunique()}")
    print(f"Feature columns: {len(df.columns) - 3}")
    print(f"\nChunks per class:\n{df['status'].value_counts()}")
    print(f"\nChunks per person: min {per_person.min()}, max {per_person.max()}, avg {per_person.mean():.1f}")