const express = require("express");
const multer = require("multer");
const axios = require("axios");
const FormData = require("form-data");
const Prediction = require("../models/Prediction");

const router = express.Router();
const upload = multer({ storage: multer.memoryStorage() });

const MODEL_SERVICE_URL =
  process.env.MODEL_SERVICE_URL || "http://127.0.0.1:8000";

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

    const modelResponse = await axios.post(
      `${MODEL_SERVICE_URL}/predict`,
      formData,
      { headers: formData.getHeaders(), timeout: 120000 },
    );

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
      // FastAPI returned an error (e.g. recording too short)
      return res
        .status(error.response.status)
        .json({ error: error.response.data.detail });
    }
    res
      .status(500)
      .json({ error: "Something went wrong processing the recording." });
  }
});

module.exports = router;
