import { useState, useRef, useEffect, useCallback } from "react";
import axios from "axios";

const BACKEND_URL = "http://localhost:5000/api";
const BAR_COUNT = 40;
const BTN_TRANSITION = "transition-all duration-200 active:scale-[0.96]";

const PASSAGE = `The North Wind and the Sun were disputing which was the stronger, when a traveler came along wrapped in a warm cloak. They agreed that the one who first succeeded in making the traveler take his cloak off should be considered stronger than the other.`;

// ---------------------------------------------------------------
// Top navigation
// ---------------------------------------------------------------
function TopNav({ tab, setTab }) {
  const tabs = [
    { id: "test", label: "Test" },
    { id: "history", label: "History" },
    { id: "about", label: "About" },
  ];
  return (
    <div className="sticky top-0 z-10 bg-white border-b border-line">
      <div className="max-w-3xl mx-auto px-6 flex items-center justify-between h-16">
        <span className="font-display text-lg font-semibold tracking-tight">
          Cadence
        </span>
        <div className="flex gap-1">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`px-4 py-2 rounded-lg text-sm font-medium ${BTN_TRANSITION} ${
                tab === t.id
                  ? "bg-accent-soft text-accent"
                  : "text-muted hover:text-ink"
              }`}
            >
              {t.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------
// Test tab — recorder + result
// ---------------------------------------------------------------
function TestTab({ onNewResult }) {
  const [phase, setPhase] = useState("idle"); // idle | recording | ready | analyzing | done
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [name, setName] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const rafRef = useRef(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const { width, height } = canvas;
    ctx.clearRect(0, 0, width, height);

    const barWidth = width / BAR_COUNT;
    const gap = barWidth * 0.35;
    let heights = new Array(BAR_COUNT).fill(0.08);

    if (phase === "recording" && analyserRef.current) {
      const bufferLength = analyserRef.current.frequencyBinCount;
      const data = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(data);
      const step = Math.floor(bufferLength / BAR_COUNT);
      heights = heights.map((_, i) => Math.max(0.08, data[i * step] / 255));
    } else if (phase === "analyzing") {
      const t = Date.now() / 250;
      heights = heights.map(
        (_, i) => 0.25 + 0.2 * Math.abs(Math.sin(t + i * 0.5)),
      );
    } else if (phase === "idle") {
      const t = Date.now() / 1400;
      heights = heights.map(
        (_, i) => 0.08 + 0.04 * Math.abs(Math.sin(t + i * 0.3)),
      );
    }

    heights.forEach((h, i) => {
      const barHeight = h * height;
      const x = i * barWidth + gap / 2;
      const y = (height - barHeight) / 2;
      ctx.fillStyle = phase === "recording" ? "#2dd4bf" : "#1e3a3a";
      ctx.fillRect(x, y, barWidth - gap, barHeight);
    });

    rafRef.current = requestAnimationFrame(draw);
  }, [phase]);

  useEffect(() => {
    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  const startRecording = async () => {
    setError(null);
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      audioCtxRef.current = audioCtx;
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((track) => track.stop());
        audioCtxRef.current?.close();
        setPhase("ready");
      };

      mediaRecorder.start();
      setPhase("recording");
    } catch (err) {
      setError(
        "Microphone access was blocked. Allow microphone permission and try again.",
      );
    }
  };

  const stopRecording = () => mediaRecorderRef.current?.stop();

  const submitRecording = async () => {
    if (!audioBlob) return;
    setPhase("analyzing");
    setError(null);

    const formData = new FormData();
    formData.append("file", audioBlob, "recording.webm");
    formData.append("name", name || "Anonymous");

    try {
      const response = await axios.post(`${BACKEND_URL}/predict`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(response.data);
      setPhase("done");
      onNewResult?.();
    } catch (err) {
      setError(
        err.response?.data?.error ||
          "Something went wrong processing the recording.",
      );
      setPhase("ready");
    }
  };

  const reset = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setResult(null);
    setError(null);
    setPhase("idle");
  };

  const isPD = result?.prediction === "Parkinson's";

  return (
    <div>
      <div className="mb-8">
        <h1 className="font-display text-2xl font-semibold tracking-tight">
          Listen to how you speak
        </h1>
        <p className="text-muted mt-2 leading-relaxed">
          A short reading test that looks for early vocal signs of Parkinson's
          disease, built as an academic screening demo — not a medical
          diagnosis.
        </p>
      </div>

      <div className="border border-line bg-white rounded-xl p-5 mb-6">
        <p className="text-sm font-medium text-ink mb-2">
          Read this passage aloud
        </p>
        <p className="text-muted leading-relaxed text-[0.95rem]">{PASSAGE}</p>
      </div>

      <input
        type="text"
        placeholder="Your name (optional)"
        value={name}
        onChange={(e) => setName(e.target.value)}
        disabled={phase === "recording" || phase === "analyzing"}
        className="w-full px-4 py-2.5 rounded-lg border border-line bg-white mb-6
                   text-ink placeholder:text-muted focus:outline-none focus:ring-2
                   focus:ring-accent/40 focus:border-accent transition-all duration-200"
      />

      <div className="bg-panel rounded-2xl p-6 mb-6">
        <canvas
          ref={canvasRef}
          width={560}
          height={96}
          className="w-full h-24 block"
        />

        <div className="flex items-center gap-3 mt-5">
          {phase === "idle" && (
            <button
              onClick={startRecording}
              className={`px-5 py-2.5 rounded-lg bg-accent text-white font-medium hover:bg-accent/90 ${BTN_TRANSITION}`}
            >
              Start recording
            </button>
          )}
          {phase === "recording" && (
            <button
              onClick={stopRecording}
              className={`px-5 py-2.5 rounded-lg bg-white text-panel font-medium hover:bg-white/90 flex items-center gap-2 ${BTN_TRANSITION}`}
            >
              <span className="w-2 h-2 rounded-full bg-pd" />
              Stop recording
            </button>
          )}
          {phase === "ready" && (
            <>
              <button
                onClick={submitRecording}
                className={`px-5 py-2.5 rounded-lg bg-accent text-white font-medium hover:bg-accent/90 ${BTN_TRANSITION}`}
              >
                Analyze recording
              </button>
              <button
                onClick={reset}
                className={`px-5 py-2.5 rounded-lg text-white/70 font-medium hover:text-white ${BTN_TRANSITION}`}
              >
                Record again
              </button>
            </>
          )}
          {phase === "analyzing" && (
            <p className="text-white/70 text-sm">Analyzing your recording…</p>
          )}
          {phase === "done" && (
            <button
              onClick={reset}
              className={`px-5 py-2.5 rounded-lg text-white/70 font-medium hover:text-white ${BTN_TRANSITION}`}
            >
              Run another test
            </button>
          )}
        </div>

        {audioUrl && phase === "ready" && (
          <audio src={audioUrl} controls className="w-full mt-4 h-9" />
        )}
      </div>

      {error && <p className="text-pd text-sm mb-6">{error}</p>}

      {result && phase === "done" && (
        <div className="border border-line bg-white rounded-2xl p-6 animate-fade-up">
          <p className="text-muted text-sm mb-1">Result</p>
          <h2
            className={`font-display text-3xl font-semibold mb-4 ${isPD ? "text-pd" : "text-healthy"}`}
          >
            {result.prediction}
          </h2>

          <div className="mb-4 flex justify-between text-sm text-muted">
            <span>Confidence</span>
            <span className="font-medium text-ink">{result.confidence}%</span>
          </div>

          <div className="flex justify-between text-sm text-muted pt-3 border-t border-line">
            <span>Healthy: {result.healthyProbability}%</span>
            <span>Parkinson's: {result.parkinsonsProbability}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------
// History tab
// ---------------------------------------------------------------
function HistoryTab({ refreshKey }) {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`${BACKEND_URL}/history`);
        setHistory(response.data);
        setError(null);
      } catch (err) {
        setError("Could not load recent tests.");
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [refreshKey]);

  const healthyCount = history.filter((h) => h.prediction === "Healthy").length;
  const pdCount = history.length - healthyCount;

  return (
    <div>
      <h1 className="font-display text-2xl font-semibold tracking-tight mb-2">
        Recent tests
      </h1>
      <p className="text-muted mb-6">
        Shared across everyone who has used this app.
      </p>

      {!loading && history.length > 0 && (
        <div className="flex gap-6 mb-8">
          <div>
            <p className="font-display text-2xl font-semibold">
              {history.length}
            </p>
            <p className="text-muted text-sm">Total tests</p>
          </div>
          <div>
            <p className="font-display text-2xl font-semibold text-healthy">
              {healthyCount}
            </p>
            <p className="text-muted text-sm">Healthy</p>
          </div>
          <div>
            <p className="font-display text-2xl font-semibold text-pd">
              {pdCount}
            </p>
            <p className="text-muted text-sm">Parkinson's flagged</p>
          </div>
        </div>
      )}

      {loading && <p className="text-muted text-sm">Loading…</p>}
      {error && <p className="text-pd text-sm">{error}</p>}

      {!loading && history.length === 0 && (
        <p className="text-muted text-sm">No tests recorded yet.</p>
      )}

      {!loading && history.length > 0 && (
        <div className="divide-y divide-line border-t border-line">
          {history.map((item) => {
            const pd = item.prediction === "Parkinson's";
            return (
              <div key={item._id} className="flex items-center py-3 gap-3">
                <span
                  className={`w-1 h-8 rounded-full ${pd ? "bg-pd" : "bg-healthy"}`}
                />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-sm truncate">{item.name}</p>
                  <p className="text-xs text-muted">
                    {new Date(item.createdAt).toLocaleString()}
                  </p>
                </div>
                <div className="text-right">
                  <p
                    className={`text-sm font-medium ${pd ? "text-pd" : "text-healthy"}`}
                  >
                    {item.prediction}
                  </p>
                  <p className="text-xs text-muted">{item.confidence}%</p>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------
// About tab — model methodology, for demo/viva use
// ---------------------------------------------------------------
function AboutTab() {
  const features = [
    { name: "Pitch", detail: "Fundamental frequency (mean, max, min)" },
    { name: "Jitter", detail: "Cycle-to-cycle frequency variation" },
    { name: "Shimmer", detail: "Cycle-to-cycle amplitude variation" },
    { name: "MFCCs", detail: "13 coefficients describing vocal tract shape" },
    { name: "Speech rate", detail: "Estimated syllables per second" },
  ];

  return (
    <div>
      <h1 className="font-display text-2xl font-semibold tracking-tight mb-2">
        About this model
      </h1>
      <p className="text-muted mb-8 leading-relaxed">
        Trained on the MDVR-KCL dataset (King's College London) — 37
        participants reading a fixed passage aloud, recorded on a smartphone.
      </p>

      <div className="mb-8">
        <p className="text-sm font-medium text-ink mb-3">
          Acoustic features analyzed
        </p>
        <div className="divide-y divide-line border-t border-b border-line">
          {features.map((f) => (
            <div key={f.name} className="flex justify-between py-3">
              <span className="text-sm font-medium">{f.name}</span>
              <span className="text-sm text-muted text-right">{f.detail}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="mb-8">
        <p className="text-sm font-medium text-ink mb-3">
          Validated performance
        </p>
        <div className="grid grid-cols-2 gap-4">
          {[
            { label: "Accuracy", value: "78.9%" },
            { label: "Sensitivity", value: "75.0%" },
            { label: "Specificity", value: "80.0%" },
            { label: "ROC-AUC", value: "0.817" },
          ].map((m) => (
            <div key={m.label} className="border border-line rounded-lg p-4">
              <p className="font-display text-xl font-semibold">{m.value}</p>
              <p className="text-muted text-sm">{m.label}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="border border-line bg-white rounded-xl p-5">
        <p className="text-sm font-medium text-ink mb-2">
          A note on limitations
        </p>
        <p className="text-muted text-sm leading-relaxed">
          This model was trained on 37 people — a small sample for a clinical
          task. Results should be read as a screening demonstration of the
          methodology, not a diagnostic result. Consult a medical professional
          for an actual evaluation.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------
// App shell
// ---------------------------------------------------------------
function App() {
  const [tab, setTab] = useState("test");
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  return (
    <div className="min-h-screen bg-bg text-ink">
      <TopNav tab={tab} setTab={setTab} />
      <div className="max-w-3xl mx-auto px-6 py-10">
        {tab === "test" && (
          <div key="test" className="animate-fade-up">
            <TestTab onNewResult={() => setHistoryRefreshKey((k) => k + 1)} />
          </div>
        )}
        {tab === "history" && (
          <div key="history" className="animate-fade-up">
            <HistoryTab refreshKey={historyRefreshKey} />
          </div>
        )}
        {tab === "about" && (
          <div key="about" className="animate-fade-up">
            <AboutTab />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
