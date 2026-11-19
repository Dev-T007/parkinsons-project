"""
main.py
FastAPI microservice: accepts an audio file, extracts features,
returns a Parkinson's/Healthy prediction with confidence.

Model v2 (chunk model):
  - the recording is cut into 8-second chunks (4-second overlap), exactly like training
  - the same 25 features are extracted from every chunk
  - the SVM predicts each chunk, and the chunk probabilities are averaged
  - the tuned threshold (0.29) decides Parkinson's vs Healthy
"""
import os
import tempfile
import joblib
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from feature_extraction import extract_all_features

app = FastAPI(title="Parkinson's Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR = "model"
model = joblib.load(os.path.join(MODEL_DIR, "mdvr_best_model_3.joblib"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "mdvr_feature_columns_3.joblib"))
THRESHOLD = float(joblib.load(os.path.join(MODEL_DIR, "mdvr_threshold_3.joblib")))

MIN_DURATION_SEC = 8  # reject recordings too short to be meaningful
CHUNK_SEC = 8.0       # must match 10_extract_chunks.py
HOP_SEC = 4.0         # must match 10_extract_chunks.py


def chunk_probabilities(wav_path):
    """Cut the recording into chunks (same as training) and return P(Parkinson's) per chunk."""
    y, sr = librosa.load(wav_path, sr=None)
    chunk_len, hop_len = int(CHUNK_SEC * sr), int(HOP_SEC * sr)

    rows = []
    for start in range(0, max(1, len(y) - chunk_len + 1), hop_len):
        seg = y[start:start + chunk_len]
        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp_path, seg, sr)
            feats = extract_all_features(tmp_path)
        except Exception:
            continue                                   # silent / unusable chunk
        finally:
            os.remove(tmp_path)
        if feats["Fo_mean"] == 0 or any(np.isnan(v) for v in feats.values()):
            continue                                   # no usable voice in this chunk
        rows.append([feats[col] for col in feature_cols])

    if not rows:
        return np.array([])
    return model.predict_proba(np.array(rows))[:, 1]


def to_display_probability(p_raw):
    """
    Rescale the raw probability so the tuned threshold sits at 50%.
    raw = THRESHOLD -> 50%, raw = 0 -> 0%, raw = 1 -> 100%.
    This keeps the UI consistent: whichever class is shown is always >= 50%.
    """
    if p_raw < THRESHOLD:
        return 0.5 * p_raw / THRESHOLD
    return 0.5 + 0.5 * (p_raw - THRESHOLD) / (1 - THRESHOLD)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Parkinson's voice detection API is running", "model": "v2-chunk-svm"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file to a temp location
    suffix = os.path.splitext(file.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        tmp_in_path = tmp_in.name

    tmp_wav_path = tmp_in_path.rsplit(".", 1)[0] + "_converted.wav"

    try:
        # Convert to WAV (browser audio is typically webm/ogg, not wav)
        audio = AudioSegment.from_file(tmp_in_path)
        audio.export(tmp_wav_path, format="wav")

        duration_sec = len(audio) / 1000
        if duration_sec < MIN_DURATION_SEC:
            raise HTTPException(
                status_code=400,
                detail=f"Recording too short ({duration_sec:.1f}s). Please read the full passage, at least {MIN_DURATION_SEC}s."
            )

        # Chunk -> predict each chunk -> average
        probs = chunk_probabilities(tmp_wav_path)
        if len(probs) == 0:
            raise HTTPException(
                status_code=400,
                detail="Could not detect a clear voice in the recording. Please record again in a quiet place."
            )

        p_raw = float(np.mean(probs))
        prediction = 1 if p_raw >= THRESHOLD else 0
        p_display = to_display_probability(p_raw)

        label = "Parkinson's" if prediction == 1 else "Healthy"
        confidence = p_display if prediction == 1 else 1 - p_display

        return {
            "prediction": label,
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                "healthy": round((1 - p_display) * 100, 1),
                "parkinsons": round(p_display * 100, 1),
            },
            "duration_seconds": round(duration_sec, 1),
            "chunks_analyzed": int(len(probs)),
            "raw_parkinsons_probability": round(p_raw, 3),
            "threshold": THRESHOLD,
            "disclaimer": "This is a screening demo for an academic project, not a medical diagnosis."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    finally:
        # Clean up temp files
        for path in [tmp_in_path, tmp_wav_path]:
            if os.path.exists(path):
                os.remove(path)