const express = require("express");
const Prediction = require("../models/Prediction");

const router = express.Router();

router.get("/history", async (req, res) => {
  try {
    const predictions = await Prediction.find()
      .sort({ createdAt: -1 })
      .limit(100);
    res.json(predictions);
  } catch (error) {
    res.status(500).json({ error: "Failed to fetch history." });
  }
});

module.exports = router;
