import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import "./App.css";

const API_URL = "http://127.0.0.1:5000";

// Comparison Slider Component
function ComparisonSlider({ originalImage, heatmapImage }) {
  const [sliderPosition, setSliderPosition] = useState(50);

  const handleMouseMove = (e) => {
    const container = e.currentTarget;
    const rect = container.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const newPosition = (x / rect.width) * 100;
    setSliderPosition(Math.max(0, Math.min(100, newPosition)));
  };

  return (
    <div className="comparison-slider" onMouseMove={handleMouseMove}>
      <div className="comparison-container">
        <div className="comparison-img comparison-original">
          <img src={originalImage} alt="original" />
          <span className="comparison-label">Original</span>
        </div>
        <div
          className="comparison-img comparison-heatmap"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <img src={heatmapImage} alt="heatmap" />
          <span className="comparison-label">AI Explanation</span>
        </div>
        <div
          className="comparison-slider-handle"
          style={{ left: `${sliderPosition}%` }}
        >
          <div className="slider-arrow" />
        </div>
      </div>
    </div>
  );
}

// Robustness Chart Component
function RobustnessChart({ results }) {
  const chartData = results.map((r) => ({
    name: r.perturbation.replace(/_/g, " "),
    confidence: r.confidence,
    time: r.inference_time_ms,
  }));

  return (
    <div className="chart-container">
      <h4>Robustness Test Results</h4>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
          <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
          <YAxis yAxisId="left" />
          <YAxis yAxisId="right" orientation="right" />
          <Tooltip />
          <Legend />
          <Bar
            yAxisId="left"
            dataKey="confidence"
            fill="#6366f1"
            name="Confidence (%)"
          />
          <Bar yAxisId="right" dataKey="time" fill="#ec4899" name="Time (ms)" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// Metrics Chart Component
function MetricsChart({ metrics }) {
  const metricsData = Object.entries(metrics.evaluation_metrics).map(
    ([key, value]) => ({
      name: key.replace(/_/g, " ").toUpperCase(),
      value: Math.round(value * 100),
    })
  );

  const classData = Object.entries(metrics.per_class_metrics).map(
    ([className, classMetrics]) => ({
      name: className.toUpperCase(),
      precision: Math.round(classMetrics.precision * 100),
      recall: Math.round(classMetrics.recall * 100),
      f1: Math.round(classMetrics.f1_score * 100),
    })
  );

  const confusionData = [
    { name: "True Negative", value: metrics.confusion_matrix[0][0] },
    { name: "False Positive", value: metrics.confusion_matrix[0][1] },
    { name: "False Negative", value: metrics.confusion_matrix[1][0] },
    { name: "True Positive", value: metrics.confusion_matrix[1][1] },
  ];

  const colors = ["#6366f1", "#ec4899", "#f59e0b", "#10b981"];

  return (
    <div className="metrics-charts">
      <div className="chart-container">
        <h4>Overall Performance Metrics</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={metricsData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
            <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Bar dataKey="value" fill="#6366f1" name="Score (%)" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h4>Per-Class Performance</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(99,102,241,0.1)" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="precision" fill="#6366f1" name="Precision" />
            <Bar dataKey="recall" fill="#ec4899" name="Recall" />
            <Bar dataKey="f1" fill="#f59e0b" name="F1 Score" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="chart-container">
        <h4>Confusion Matrix</h4>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={confusionData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, value }) => `${name}: ${value}`}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {confusionData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={colors[index]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [image, setImage]     = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [heatmap, setHeatmap] = useState(null);
  const [showExplanation, setShowExplanation] = useState(false);
  const [useComparisonSlider, setUseComparisonSlider] = useState(false);
  const [robustnessResults, setRobustnessResults] = useState(null);
  const [showRobustness, setShowRobustness] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [showMetrics, setShowMetrics] = useState(false);
  const [activeTab, setActiveTab] = useState('analysis'); // 'analysis', 'robustness', 'metrics'

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
    setShowExplanation(false);
    setRobustnessResults(null);
    setShowRobustness(false);
    setActiveTab('analysis');
  };

  const runRobustnessTest = async () => {
    if (!image) return;
    setLoading(true);
    const fd = new FormData();
    fd.append("file", image);
    try {
      const res = await axios.post(`${API_URL}/robustness-test`, fd);
      setRobustnessResults(res.data);
      setActiveTab('robustness');
    } catch {
      alert("Cannot connect to backend! Make sure it is running on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const fetchMetrics = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_URL}/metrics`);
      setMetrics(res.data.metrics);
      setActiveTab('metrics');
    } catch {
      alert("Cannot connect to backend! Make sure it is running on port 8000.");
    } finally {
      setLoading(false);
    }
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

        {/* Tab Navigation */}
        <div className="tab-navigation">
          <button
            className={`tab-button ${activeTab === 'analysis' ? 'active' : ''}`}
            onClick={() => setActiveTab('analysis')}
          >
            Image Analysis
          </button>
          <button
            className={`tab-button ${activeTab === 'robustness' ? 'active' : ''}`}
            onClick={() => setActiveTab('robustness')}
          >
            Robustness Test
          </button>
          <button
            className={`tab-button ${activeTab === 'metrics' ? 'active' : ''}`}
            onClick={() => fetchMetrics()}
          >
            Model Metrics
          </button>
        </div>

        {/* Analysis Tab */}
        {activeTab === 'analysis' && (
          <>
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

            {/* Image preview with overlay */}
            {preview && (
              <div className="image-preview-container">
                <div className="image-card">
                  <div className="image-label">
                    <span>Image Analysis</span>
                    {result && heatmap && (
                      <div className="heatmap-controls">
                        <button
                          className={`explanation-toggle ${showExplanation && !useComparisonSlider ? 'active' : ''}`}
                          onClick={() => {
                            setShowExplanation(true);
                            setUseComparisonSlider(false);
                          }}
                        >
                          Overlay
                        </button>
                        <button
                          className={`explanation-toggle ${useComparisonSlider ? 'active' : ''}`}
                          onClick={() => {
                            setShowExplanation(false);
                            setUseComparisonSlider(true);
                          }}
                        >
                          Compare
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="image-wrapper">
                    {useComparisonSlider && heatmap ? (
                      <ComparisonSlider
                        originalImage={preview}
                        heatmapImage={heatmap}
                      />
                    ) : (
                      <>
                        <img
                          src={preview}
                          alt="original"
                          className="original-image"
                        />
                        {showExplanation && heatmap && (
                          <img
                            src={heatmap}
                            alt="heatmap overlay"
                            className="heatmap-overlay"
                          />
                        )}
                      </>
                    )}
                  </div>
                </div>
              </div>
            )}

            {/* Action buttons */}
            {preview && !loading && (
              <div className="action-buttons">
                {!result && (
                  <button className="analyze-btn" onClick={analyze}>
                    Analyze Image
                  </button>
                )}
                <button className="robustness-btn" onClick={runRobustnessTest}>
                  Run Robustness Test
                </button>
              </div>
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
                  
                  <div className="confidence-section">
                    <div className="conf-label-row">
                      <span>Confidence Score</span>
                      <span className="conf-percentage">{result.confidence}%</span>
                    </div>
                    <div className="confidence-meter">
                      <div className="conf-track">
                        <div
                          className="conf-fill"
                          style={{ width: `${result.confidence}%` }}
                        />
                      </div>
                      <div className="confidence-markers">
                        <span>0%</span>
                        <span>50%</span>
                        <span>100%</span>
                      </div>
                    </div>
                  </div>

                  <div className="result-stats">
                    <div className="stat-badge">
                      <span className="stat-icon">⚡</span>
                      <span>{result.inference_time_ms}ms</span>
                    </div>
                    <div className="stat-badge">
                      <span className="stat-icon">🎯</span>
                      <span>{result.model_accuracy}% Accuracy</span>
                    </div>
                  </div>

                  <p className="result-msg">{result.message}</p>
                </div>
                <button className="reset-btn" onClick={reset}>
                  Analyze Another Image
                </button>
              </div>
            )}
          </>
        )}

        {/* Robustness Tab */}
        {activeTab === 'robustness' && (
          <div className="robustness-tab">
            {!robustnessResults && !loading && (
              <div className="robustness-intro">
                <h3>Model Robustness Testing</h3>
                <p>Test how well our model performs under various image perturbations and transformations.</p>
                {preview ? (
                  <button className="robustness-btn" onClick={runRobustnessTest}>
                    Run Robustness Test
                  </button>
                ) : (
                  <p className="upload-prompt">Please upload an image first in the Analysis tab.</p>
                )}
              </div>
            )}

            {loading && (
              <div className="loading-wrap">
                <div className="spinner"/>
                <p>Running robustness tests...</p>
              </div>
            )}

            {robustnessResults && (
              <div className="robustness-results">
                <div className="robustness-summary">
                  <h3>Robustness Summary</h3>
                  <div className="summary-stats">
                    <div className="stat-item">
                      <span className="stat-value">{robustnessResults.summary.robustness_score * 100}%</span>
                      <span className="stat-label">Consistency</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-value">{robustnessResults.summary.successful_tests}</span>
                      <span className="stat-label">Tests Passed</span>
                    </div>
                    <div className="stat-item">
                      <span className="stat-value">{robustnessResults.summary.average_inference_time_ms}ms</span>
                      <span className="stat-label">Avg Time</span>
                    </div>
                  </div>
                  <p className="summary-message">{robustnessResults.message}</p>
                </div>

                <RobustnessChart results={robustnessResults.results} />

                <div className="robustness-details">
                  <h4>Detailed Test Results</h4>
                  <div className="test-results-grid">
                    {robustnessResults.results.map((test, index) => (
                      <div key={index} className={`test-result-card ${test.error ? 'error' : ''}`}>
                        <div className="test-name">{test.perturbation.replace('_', ' ').toUpperCase()}</div>
                        {test.error ? (
                          <div className="test-error">Error: {test.error}</div>
                        ) : (
                          <>
                            <div className="test-prediction">{test.prediction.toUpperCase()}</div>
                            <div className="test-confidence">{test.confidence}%</div>
                            <div className="test-time">{test.inference_time_ms}ms</div>
                          </>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Metrics Tab */}
        {activeTab === 'metrics' && (
          <div className="metrics-tab">
            {loading && (
              <div className="loading-wrap">
                <div className="spinner"/>
                <p>Loading model metrics...</p>
              </div>
            )}

            {metrics && (
              <div className="metrics-content">
                <h3>Model Evaluation Metrics</h3>

                <MetricsChart metrics={metrics} />

                <div className="metrics-section">
                  <h4>Model Information</h4>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">Architecture</span>
                      <span className="metric-value">{metrics.model_info.architecture}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Parameters</span>
                      <span className="metric-value">{metrics.model_info.num_parameters.toLocaleString()}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Input Size</span>
                      <span className="metric-value">{metrics.model_info.input_size}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Device</span>
                      <span className="metric-value">{metrics.performance_stats.device}</span>
                    </div>
                  </div>
                </div>

                <div className="metrics-section">
                  <h4>Overall Performance</h4>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">Accuracy</span>
                      <span className="metric-value">{(metrics.evaluation_metrics.accuracy * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">F1 Score (Macro)</span>
                      <span className="metric-value">{(metrics.evaluation_metrics.f1_macro * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">AUC-ROC</span>
                      <span className="metric-value">{(metrics.evaluation_metrics.auc_roc * 100).toFixed(1)}%</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Avg Inference Time</span>
                      <span className="metric-value">{metrics.performance_stats.average_inference_time_ms}ms</span>
                    </div>
                  </div>
                </div>

                <div className="metrics-section">
                  <h4>Per-Class Performance</h4>
                  <div className="class-metrics">
                    {Object.entries(metrics.per_class_metrics).map(([className, classMetrics]) => (
                      <div key={className} className="class-metric-card">
                        <h5>{className.toUpperCase()}</h5>
                        <div className="class-stats">
                          <div className="class-stat">
                            <span>Precision</span>
                            <span>{(classMetrics.precision * 100).toFixed(1)}%</span>
                          </div>
                          <div className="class-stat">
                            <span>Recall</span>
                            <span>{(classMetrics.recall * 100).toFixed(1)}%</span>
                          </div>
                          <div className="class-stat">
                            <span>F1 Score</span>
                            <span>{(classMetrics.f1_score * 100).toFixed(1)}%</span>
                          </div>
                          <div className="class-stat">
                            <span>Samples</span>
                            <span>{classMetrics.support}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="metrics-section">
                  <h4>Training Information</h4>
                  <div className="metrics-grid">
                    <div className="metric-item">
                      <span className="metric-label">Epochs Trained</span>
                      <span className="metric-value">{metrics.training_info.epochs_trained}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Best Epoch</span>
                      <span className="metric-value">{metrics.training_info.best_epoch}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Final Loss</span>
                      <span className="metric-value">{metrics.training_info.final_loss}</span>
                    </div>
                    <div className="metric-item">
                      <span className="metric-label">Early Stopping</span>
                      <span className="metric-value">{metrics.training_info.early_stopping ? 'Yes' : 'No'}</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
