import React from "react";
import "./CharacterGrid.css";

export default function CharacterGrid({ characters }) {
  if (!characters || characters.length === 0) return null;

  const getColor = (conf) => {
    if (conf >= 0.9) return "#4caf7d";
    if (conf >= 0.75) return "#d4a853";
    return "#e05c5c";
  };

  return (
    <div className="char-grid">
      <h3 className="char-grid-title">Character Analysis</h3>
      <div className="char-grid-items">
        {characters.map((item, i) => {
          const pct = Math.round(item.confidence * 100);
          const color = getColor(item.confidence);
          return (
            <div
              className="char-card"
              key={i}
              style={{ animationDelay: `${i * 60}ms` }}
            >
              <div className="char-glyph">{item.char}</div>
              <div className="char-conf-bar-wrap">
                <div
                  className="char-conf-bar"
                  style={{ width: `${pct}%`, background: color }}
                />
              </div>
              <div className="char-pct" style={{ color }}>{pct}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
