import React, { useState } from "react";

export default function ResultDisplay({ result, onReset }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(result.text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div style={{ maxWidth: 640, margin: "0 auto" }}>
      <div style={{
        background: "#f0fdf4", border: "1px solid #86efac",
        borderRadius: 12, padding: "16px 20px", marginBottom: 20,
        display: "flex", alignItems: "center", gap: 8,
      }}>
        <span style={{ color: "#16a34a", fontWeight: 500 }}>✓ Recognition Complete</span>
        {result.processing_time_ms && (
          <span style={{ marginLeft: "auto", color: "#888", fontSize: 13 }}>
            {result.processing_time_ms}ms
          </span>
        )}
      </div>

      <div style={{
        border: "1px solid #e5e7eb", borderRadius: 12,
        padding: 20, marginBottom: 20,
      }}>
        <div style={{ fontSize: 12, color: "#888", marginBottom: 8 }}>Recognized Text</div>
        <p style={{
          fontSize: 22, lineHeight: 1.7, margin: "0 0 16px",
          fontFamily: "serif", whiteSpace: "pre-wrap",
          minHeight: 40,
        }}>
          {result.text || "No text detected"}
        </p>
        <button
          onClick={handleCopy}
          style={{
            padding: "8px 20px", borderRadius: 8, border: "1px solid #d1d5db",
            background: copied ? "#16a34a" : "#fff",
            color: copied ? "#fff" : "#374151",
            cursor: "pointer", fontSize: 14, fontWeight: 500,
          }}
        >
          {copied ? "✓ Copied!" : "Copy Text"}
        </button>
      </div>

      <div style={{ display: "flex", gap: 16, marginBottom: 20 }}>
        {result.avg_confidence != null && (
          <div style={{
            flex: 1, border: "1px solid #e5e7eb", borderRadius: 12,
            padding: "16px", textAlign: "center",
          }}>
            <div style={{ fontSize: 24, fontWeight: 600 }}>{Math.round(result.avg_confidence)}%</div>
            <div style={{ fontSize: 12, color: "#888" }}>Avg. Confidence</div>
          </div>
        )}
        {result.char_count != null && (
          <div style={{
            flex: 1, border: "1px solid #e5e7eb", borderRadius: 12,
            padding: "16px", textAlign: "center",
          }}>
            <div style={{ fontSize: 24, fontWeight: 600 }}>{result.char_count}</div>
            <div style={{ fontSize: 12, color: "#888" }}>Characters</div>
          </div>
        )}
        {result.line_count != null && (
          <div style={{
            flex: 1, border: "1px solid #e5e7eb", borderRadius: 12,
            padding: "16px", textAlign: "center",
          }}>
            <div style={{ fontSize: 24, fontWeight: 600 }}>{result.line_count}</div>
            <div style={{ fontSize: 12, color: "#888" }}>Lines</div>
          </div>
        )}
      </div>

      {result.characters && result.characters.length > 0 && (
        <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 20, marginBottom: 20 }}>
          <div style={{ fontSize: 14, fontWeight: 500, marginBottom: 12 }}>Character Analysis</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
            {result.characters.map((ch, i) => {
              const pct   = Math.round(ch.confidence);
              const color = pct >= 90 ? "#16a34a" : pct >= 70 ? "#ca8a04" : "#dc2626";
              return (
                <div key={i} style={{
                  border: `1px solid ${color}22`,
                  borderRadius: 8, padding: "6px 10px",
                  background: `${color}08`,
                  textAlign: "center", minWidth: 48,
                }}>
                  <div style={{ fontSize: 18, fontFamily: "serif" }}>{ch.letter}</div>
                  <div style={{ fontSize: 11, color }}>{pct}%</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      <button
        onClick={onReset}
        style={{
          padding: "10px 24px", borderRadius: 8, border: "1px solid #d1d5db",
          background: "#fff", cursor: "pointer", fontSize: 14,
        }}
      >
        ← Recognize Another
      </button>
    </div>
  );
}