require("dotenv").config();
const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");

const predictRoute = require("./routes/predict");
const historyRoute = require("./routes/history");

const app = express();
app.use(cors());
app.use(express.json());

app.use("/api", predictRoute);
app.use("/api", historyRoute);

app.get("/", (req, res) => {
  res.json({ status: "ok", message: "Parkinson's app backend running" });
});

const PORT = process.env.PORT || 5000;

mongoose
  .connect(process.env.MONGODB_URI)
  .then(() => {
    console.log("Connected to MongoDB");
    app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
  })
  .catch((err) => {
    console.error("MongoDB connection error:", err.message);
  });
