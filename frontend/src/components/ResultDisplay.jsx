import React, { useState } from "react";
import CharacterGrid from "./CharacterGrid";
import "./ResultDisplay.css";

export default function ResultDisplay({ result, onReset }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(result.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const avgConf = result.characters
    ? Math.round(
        (result.characters.reduce((s, c) => s + c.confidence, 0) /
          result.characters.length) * 100
      )
    : null;

  return (
    <div className="result-wrapper">
      <div className="result-header">
        <div className="result-badge">
          <span className="badge-dot" />
          Recognition Complete
        </div>
        {result.processing_time_ms && (
          <span className="result-time">{result.processing_time_ms}ms</span>
        )}
      </div>

      <div className="result-text-box">
        <div className="result-label">Recognized Text</div>
        <p className="result-text">{result.text}</p>
        <button className={`copy-btn ${copied ? "copied" : ""}`} onClick={handleCopy}>
          {copied ? "✓ Copied!" : "Copy Text"}
        </button>
      </div>

      {avgConf !== null && (
        <div className="result-stats">
          <div className="stat-item">
            <span className="stat-value">{avgConf}%</span>
            <span className="stat-label">Avg. Confidence</span>
          </div>
          {result.characters && (
            <div className="stat-item">
              <span className="stat-value">{result.characters.length}</span>
              <span className="stat-label">Characters</span>
            </div>
          )}
          {result.processing_time_ms && (
            <div className="stat-item">
              <span className="stat-value">{result.processing_time_ms}ms</span>
              <span className="stat-label">Process Time</span>
            </div>
          )}
        </div>
      )}

      <CharacterGrid characters={result.characters} />

      <button className="reset-btn" onClick={onReset}>← Recognize Another</button>
    </div>
  );
}
