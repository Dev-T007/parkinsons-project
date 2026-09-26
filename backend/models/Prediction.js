const mongoose = require("mongoose");

const predictionSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      default: "Anonymous",
      trim: true,
    },
    prediction: {
      type: String,
      enum: ["Healthy", "Parkinson's"],
      required: true,
    },
    confidence: {
      type: Number,
      required: true,
    },
    healthyProbability: {
      type: Number,
      required: true,
    },
    parkinsonsProbability: {
      type: Number,
      required: true,
    },
    durationSeconds: {
      type: Number,
    },
  },
  { timestamps: true },
);

module.exports = mongoose.model("Prediction", predictionSchema);
