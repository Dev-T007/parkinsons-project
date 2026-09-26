# Cadence — Voice-Based Parkinson's Disease Screening

An AI-based system that analyzes short voice recordings to identify early
vocal biomarkers associated with Parkinson's disease. Built as an academic
minor project.

**⚠️ This is an academic screening demo, not a medical diagnostic tool.**

## Architecture

- **frontend/** — React + Tailwind UI. Records voice via the Web Audio API,
  shows prediction results and a shared history of past tests.
- **backend/** — Node/Express REST API. Receives audio, forwards it to the
  model service, saves results to MongoDB, serves prediction history.
- **model-service/** — Python FastAPI microservice. Extracts acoustic
  features from audio and runs the trained ML model.
- **ml-model/** — Model development: data exploration, training, tuning,
  evaluation, and explainability (not part of the running app itself).

## The model

Trained on the [MDVR-KCL dataset](https://zenodo.org/records/2867216)
(King's College London) — 37 participants (16 Parkinson's, 21 healthy)
reading a fixed passage aloud, recorded on a smartphone.

**Features extracted:** pitch (F0), jitter, shimmer, 13 MFCCs, speech rate
— 25 features total, covering acoustic, spectral, and prosodic biomarkers.

**Best model:** Random Forest (tuned)
| Metric | Score |
|---|---|
| Accuracy | 78.9% |
| Sensitivity | 75.0% |
| Specificity | 80.0% |
| ROC-AUC | 0.817 |

Validated using `StratifiedKFold` cross-validation. See `ml-model/reports/`
for confusion matrix and SHAP explainability plots.

## Running locally

Requires: Python 3.10+, Node.js, ffmpeg, and a MongoDB connection string
(Atlas or local).

**1. Model service**
```bash
cd model-service
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**2. Backend**
```bash
cd backend
npm install
# create a .env file — see backend/README section below
npm run dev
```

**3. Frontend**
```bash
cd frontend
npm install
npm run dev
```

All three must be running simultaneously for the app to work end-to-end.

## Environment variables (backend/.env)

MONGODB_URI=your_mongodb_connection_string
PORT=5000
MODEL_SERVICE_URL=http://127.0.0.1:8000


## Limitations

- Trained on only 37 participants — small sample for a clinical task
- Specificity (80%) is stronger than earlier iterations, but results should
  be read as a methodology demonstration, not a diagnostic result
- Model performance on live recordings can vary based on microphone
  quality and recording conditions

## Tech stack

Python, scikit-learn, XGBoost, SHAP, Librosa, Parselmouth, FastAPI,
Node.js, Express, MongoDB, React, Tailwind CSS