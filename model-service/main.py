"""
main.py
FastAPI microservice: accepts an audio file, extracts features,
returns a Parkinson's/Healthy prediction with confidence.
"""
import os
import tempfile
import joblib
import numpy as np
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
model = joblib.load(os.path.join(MODEL_DIR, "mdvr_best_model.joblib"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "mdvr_feature_columns.joblib"))

MIN_DURATION_SEC = 8  # reject recordings too short to be meaningful

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Parkinson's voice detection API is running"}

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

        # Extract features and predict
        feats = extract_all_features(tmp_wav_path)
        X = np.array([[feats[col] for col in feature_cols]])

        prediction = int(model.predict(X)[0])
        probability = model.predict_proba(X)[0]

        label = "Parkinson's" if prediction == 1 else "Healthy"
        confidence = float(probability[prediction] * 100)

        return {
            "prediction": label,
            "confidence": round(confidence, 1),
            "probabilities": {
                "healthy": round(float(probability[0] * 100), 1),
                "parkinsons": round(float(probability[1] * 100), 1),
            },
            "duration_seconds": round(duration_sec, 1),
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