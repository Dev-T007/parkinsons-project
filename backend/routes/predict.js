const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const Prediction = require("../models/Prediction");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

const MODEL_SERVICE_URL =
  process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8000";

// Render's free tier spins the model service down after ~15 min idle.
// The first request after that has to cold-boot a fresh container
// (FastAPI + librosa + scipy + scikit-learn + xgboost), which can take
// 30-60s. Render's proxy does NOT queue that first request — it bounces
// it with a 502/503 immediately instead of waiting. So we retry a few
// times with a short delay in between, which covers a full cold boot
// without the user ever seeing an error.
async function callModelService(formData, attempt = 1, maxAttempts = 5) {
  try {
    return await axios.post(`${MODEL_SERVICE_URL}/predict`, formData, {
      headers: formData.getHeaders(),
      timeout: 60000,
    });
  } catch (err) {
    const status = err.response?.status;
    const isColdStart = status === 502 || status === 503 || !err.response;

    if (isColdStart && attempt < maxAttempts) {
      console.log(
        `Model service not ready (attempt ${attempt}/${maxAttempts}, status ${status || "no response"}). Retrying in 10s...`,
      );
      await new Promise((resolve) => setTimeout(resolve, 10000));
      return callModelService(formData, attempt + 1, maxAttempts);
    }
    throw err;
  }
}

router.post("/predict", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No audio file uploaded." });
    }

    // Forward the file to the FastAPI model service
    const formData = new FormData();
    formData.append("file", req.file.buffer, {
      filename: req.file.originalname || "recording.webm",
      contentType: req.file.mimetype,
    });

    const modelResponse = await callModelService(formData);
    const result = modelResponse.data;

    // Save to MongoDB
    const savedPrediction = await Prediction.create({
      name: req.body.name || "Anonymous",
      prediction: result.prediction,
      confidence: result.confidence,
      healthyProbability: result.probabilities.healthy,
      parkinsonsProbability: result.probabilities.parkinsons,
      durationSeconds: result.duration_seconds,
    });

    res.json(savedPrediction);
  } catch (error) {
    console.error("Prediction error:", error.message);
    if (error.response) {
      // FastAPI returned an error (e.g. recording too short), or the
      // model service is still down after all retries.
      return res
        .status(error.response.status)
        .json({ error: error.response.data?.detail || "Prediction failed" });
    }
    res.status(502).json({ error: error.message || "Prediction failed" });
  }
});

module.exports = router;
