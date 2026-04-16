import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [image, setImage]     = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [heatmap, setHeatmap] = useState(null);

  const onDrop = useCallback((files) => {
    const file = files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setHeatmap(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { "image/*": [] }, multiple: false,
  });

  const analyze = async () => {
    if (!image) return;
    setLoading(true);
    setResult(null);
    const fd = new FormData();
    fd.append("file", image);
    try {
      const res = await axios.post(`${API_URL}/predict`, fd);
      setResult(res.data);
      if (res.data.heatmap) setHeatmap(`data:image/png;base64,${res.data.heatmap}`);
    } catch {
      alert("Cannot connect to backend! Make sure it is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setImage(null); setPreview(null);
    setResult(null); setHeatmap(null);
  };

  return (
    <div className="app">

      {/* Navbar */}
      <nav className="navbar">
        <div className="brand">
          <div className="brand-dot" />
          DeepGuard AI
        </div>
        <span className="nav-badge">EfficientNet-B0</span>
      </nav>

      {/* Hero */}
      <div className="hero">
        <div className="hero-tag">AI Powered Detection</div>
        <h1>Deepfake Detector</h1>
        <p>
          Upload any face image and our AI model will instantly
          detect whether it is real or AI generated.
        </p>

        <div className="stats-row">
          <div className="stat">
            <div className="stat-value">97%</div>
            <div className="stat-label">Accuracy</div>
          </div>
          <div className="stat">
            <div className="stat-value">10K</div>
            <div className="stat-label">Trained On</div>
          </div>
          <div className="stat">
            <div className="stat-value">&lt;2s</div>
            <div className="stat-label">Detection</div>
          </div>
        </div>
      </div>

      {/* Main */}
      <div className="main-card">

        {/* Upload zone */}
        {!preview && (
          <div {...getRootProps()}
            className={`upload-area ${isDragActive ? "active" : ""}`}>
            <input {...getInputProps()} />
            <div className="upload-icon-wrap">
              <svg width="30" height="30" fill="none" stroke="#6366f1"
                strokeWidth="1.8" viewBox="0 0 24 24">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <h3>Drop your image here</h3>
            <p>or click to browse &nbsp;·&nbsp; JPG, PNG supported</p>
          </div>
        )}

        {/* Image previews */}
        {preview && (
          <div className="preview-grid">
            <div className="img-card">
              <div className="img-card-label"><span/>Original Image</div>
              <img src={preview} alt="original" />
            </div>
            <div className="img-card">
              <div className="img-card-label"><span/>Grad-CAM Heatmap</div>
              {heatmap
                ? <img src={heatmap} alt="heatmap" />
                : <div className="img-placeholder">
                    <svg width="32" height="32" fill="none" stroke="currentColor"
                      strokeWidth="1.5" viewBox="0 0 24 24">
                      <circle cx="12" cy="12" r="10"/>
                      <path d="M12 8v4l3 3"/>
                    </svg>
                    Run analysis to see heatmap
                  </div>
              }
            </div>
          </div>
        )}

        {/* Analyze button */}
        {preview && !loading && !result && (
          <button className="analyze-btn" onClick={analyze}>
            Analyze Image
          </button>
        )}

        {/* Loading */}
        {loading && (
          <div className="loading-wrap">
            <div className="spinner"/>
            <p>Analyzing with EfficientNet-B0...</p>
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="result-wrap">
            <div className={`result-card ${result.is_fake ? "fake" : "real"}`}>
              <div className="result-top">
                <div>
                  <div className="result-label">Detection Result</div>
                  <div className="result-verdict">
                    {result.is_fake ? "FAKE" : "REAL"}
                  </div>
                </div>
                <div className="result-icon">
                  {result.is_fake ? "⚠️" : "✅"}
                </div>
              </div>
              <div className="conf-label-row">
                <span>Confidence Score</span>
                <span>{result.confidence}%</span>
              </div>
              <div className="conf-track">
                <div className="conf-fill"
                  style={{ width: `${result.confidence}%` }}/>
              </div>
              <p className="result-msg">{result.message}</p>
            </div>
            <button className="reset-btn" onClick={reset}>
              Analyze Another Image
            </button>
          </div>
        )}

      </div>
    </div>
  );
}