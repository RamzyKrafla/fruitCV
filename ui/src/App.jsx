import { useState, useEffect, useCallback } from "react";
import { initModel, preprocessFileToTensor, runInference } from "./infer.js";
import "./App.css";

function App() {
  const [files, setFiles] = useState([]);
  const [session, setSession] = useState(null);
  const [loading, setLoading] = useState(false);
  const [over, setOver] = useState(false);
  const [modelStatus, setModelStatus] = useState("loading");

  // Initialize model on component mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        setModelStatus("loading");
        const modelSession = await initModel();
        setSession(modelSession);
        setModelStatus("ready");
      } catch (error) {
        console.error("Failed to load model:", error);
        setModelStatus("error");
      }
    };
    loadModel();
  }, []);

  // File handling functions
  const onDragOver = useCallback((e) => {
    e.preventDefault();
    setOver(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    setOver(false);
  }, []);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setOver(false);
    const droppedFiles = Array.from(e.dataTransfer.files);
    addFiles(droppedFiles);
  }, []);

  const onPick = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.multiple = true;
    input.accept = ".jpg,.jpeg,.png,.pdf";
    input.onchange = (e) => {
      const selectedFiles = Array.from(e.target.files);
      addFiles(selectedFiles);
    };
    input.click();
  }, []);

  const addFiles = useCallback((newFiles) => {
    const validFiles = newFiles.filter((file) => {
      const validTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "application/pdf",
      ];
      return validTypes.includes(file.type);
    });

    if (validFiles.length === 0) return;

    const fileItems = validFiles.slice(0, 12).map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      type: file.type,
      preview: file.type.startsWith("image/")
        ? URL.createObjectURL(file)
        : null,
      result: null,
      error: null,
    }));

    setFiles((prev) => [...prev, ...fileItems]);
  }, []);

  const removeFile = useCallback((id) => {
    setFiles((prev) => {
      const fileToRemove = prev.find((f) => f.id === id);
      if (fileToRemove?.preview) {
        URL.revokeObjectURL(fileToRemove.preview);
      }
      return prev.filter((f) => f.id !== id);
    });
  }, []);

  const clearAll = useCallback(() => {
    files.forEach((file) => {
      if (file.preview) {
        URL.revokeObjectURL(file.preview);
      }
    });
    setFiles([]);
  }, [files]);

  const formatBytes = (bytes) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const getFruitEmoji = (label) => {
    const emojis = {
      banana: "🍌",
      apple: "🍎",
      mango: "🥭",
    };
    return emojis[label.toLowerCase()] || "🍎";
  };

  const handleSubmit = async () => {
    if (!session || files.length === 0) return;

    setLoading(true);
    const updatedFiles = [...files];

    for (let i = 0; i < updatedFiles.length; i++) {
      const fileItem = updatedFiles[i];
      try {
        const tensor = await preprocessFileToTensor(fileItem.file);
        const result = await runInference(session, tensor);
        updatedFiles[i] = { ...fileItem, result, error: null };
      } catch (error) {
        console.error(`Error processing ${fileItem.file.name}:`, error);
        updatedFiles[i] = { ...fileItem, error: error.message, result: null };
      }
    }

    setFiles(updatedFiles);
    setLoading(false);
  };

  const getStatusText = () => {
    switch (modelStatus) {
      case "loading":
        return "Loading model...";
      case "ready":
        return "Ready";
      case "error":
        return "Model error";
      default:
        return "Unknown";
    }
  };

  return (
    <div className="container">
      {/* Header Section */}
      <div className="header">
        <h1 className="title">Fruit Classifier</h1>
        <p className="sub">
          Upload images or PDFs to classify fruits using AI. All processing
          happens in your browser.
        </p>
        <div className="status-pill">
          <div className="status-dot"></div>
          {getStatusText()}
        </div>
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div
          className={`dropzone ${over ? "over" : ""}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={onPick}
          tabIndex={0}
          role="button"
          aria-label="Upload files"
        >
          <input
            type="file"
            multiple
            accept=".jpg,.jpeg,.png,.pdf"
            style={{ display: "none" }}
            onChange={(e) => addFiles(Array.from(e.target.files))}
          />
          <div className="drop-inner">
            <svg
              className="icon"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7,10 12,15 17,10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            <div>
              <p className="drop-text">Drop files here or click to browse</p>
              <p className="drop-hint">Supports JPG, PNG, and PDF files</p>
            </div>
          </div>
        </div>

        {files.length > 0 && (
          <div className="footer">
            <button className="btn secondary" onClick={clearAll}>
              Clear all
            </button>
            <button
              className="btn primary"
              onClick={handleSubmit}
              disabled={!session || loading}
            >
              {loading ? (
                <>
                  <div className="loading-spinner"></div>
                  Classifying...
                </>
              ) : (
                "Classify Files"
              )}
            </button>
          </div>
        )}
      </div>

      {/* Results Section */}
      <div className="results-section">
        {files.length > 0 ? (
          <>
            <h2 className="results-header">Results</h2>
            <div className="files">
              {files.map((item) => (
                <div className="card" key={item.id}>
                  <div className="card-header">
                    {item.preview ? (
                      <img
                        className="thumbnail"
                        src={item.preview}
                        alt={item.file.name}
                      />
                    ) : (
                      <div className="thumbnail center">
                        <span style={{ fontSize: "1.5rem", opacity: 0.5 }}>
                          {item.type === "application/pdf" ? "📄" : "📁"}
                        </span>
                      </div>
                    )}
                    <div className="file-info">
                      <p className="file-name">{item.file.name}</p>
                      <p className="file-meta">
                        {item.type.toUpperCase()} •{" "}
                        {formatBytes(item.file.size)}
                      </p>
                    </div>
                    <button
                      className="btn danger"
                      onClick={() => removeFile(item.id)}
                      aria-label="Remove file"
                    >
                      Remove
                    </button>
                  </div>

                  {item.error && (
                    <div className="error-message">Error: {item.error}</div>
                  )}

                  {Array.isArray(item.result) && item.result.length > 0 && (
                    <div>
                      {/* Top Prediction Badge */}
                      <div
                        className={`top-prediction ${item.result[0].label.toLowerCase()}`}
                      >
                        {getFruitEmoji(item.result[0].label)}{" "}
                        {item.result[0].label}{" "}
                        {item.result[0].probability.toFixed(1)}%
                      </div>

                      {/* Probability Bars */}
                      <div className="probs">
                        {item.result.map((pred, index) => (
                          <div className="prob-row" key={pred.label}>
                            <span className="prob-label">{pred.label}</span>
                            <div className="prob-bar">
                              <div
                                className={
                                  index === 0
                                    ? `top ${pred.label.toLowerCase()}`
                                    : pred.label.toLowerCase()
                                }
                                style={{ width: `${pred.probability}%` }}
                              ></div>
                            </div>
                            <span className="prob-val">
                              {pred.probability.toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="empty-state">
            <div className="icon">🍎</div>
            <p>Upload files to see classification results</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
