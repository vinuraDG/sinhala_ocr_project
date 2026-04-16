import React, { useState, useRef, useCallback } from "react";

export default function CameraCapture({ onCapture }) {
  const [stream,      setStream]      = useState(null);
  const [facingMode,  setFacingMode]  = useState("environment");
  const [error,       setError]       = useState(null);
  const videoRef  = useRef();
  const canvasRef = useRef();

  const startCamera = useCallback(async (facing = facingMode) => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing }, audio: false,
      });
      setStream(s);
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = s;
      }, 100);
    } catch {
      setError("Camera access denied. Please allow camera permission.");
    }
  }, [facingMode]);

  const stopCamera = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    setStream(null);
  };

  const flipCamera = () => {
    stopCamera();
    const next = facingMode === "environment" ? "user" : "environment";
    setFacingMode(next);
    setTimeout(() => startCamera(next), 300);
  };

  const snapPhoto = () => {
    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const dataURL = canvas.toDataURL("image/jpeg", 0.92);
    canvas.toBlob((blob) => {
      const file = new File([blob], "captured.jpg", { type: "image/jpeg" });
      onCapture({ dataURL, file });
    }, "image/jpeg", 0.92);
    stopCamera();
  };

  if (!stream) {
    return (
      <div style={{ textAlign: "center", padding: "48px 16px" }}>
        <div style={{ fontSize: 48, marginBottom: 16 }}>📷</div>
        <p style={{ fontWeight: 500, fontSize: 18, marginBottom: 8 }}>Camera is off</p>
        <p style={{ color: "#888", marginBottom: 24 }}>
          Click below to activate your camera.
        </p>
        {error && (
          <div style={{
            background: "#fef2f2", color: "#dc2626", border: "1px solid #fca5a5",
            borderRadius: 8, padding: "10px 16px", marginBottom: 16, fontSize: 14,
          }}>
            {error}
          </div>
        )}
        <button
          onClick={() => startCamera()}
          style={{
            padding: "12px 28px", borderRadius: 10, border: "none",
            background: "#2563eb", color: "#fff", fontSize: 15,
            fontWeight: 500, cursor: "pointer",
          }}
        >
          📷 Start Camera
        </button>
      </div>
    );
  }

  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
        <video
          ref={videoRef}
          autoPlay playsInline muted
          style={{ width: "100%", maxWidth: 480, borderRadius: 12, display: "block" }}
        />
      </div>
      <canvas ref={canvasRef} style={{ display: "none" }} />
      <div style={{ display: "flex", gap: 12, justifyContent: "center", marginTop: 16 }}>
        <button
          onClick={snapPhoto}
          style={{
            padding: "12px 28px", borderRadius: 10, border: "none",
            background: "#2563eb", color: "#fff", fontSize: 15,
            fontWeight: 500, cursor: "pointer",
          }}
        >
          📸 Capture
        </button>
        <button
          onClick={flipCamera}
          style={{
            padding: "12px 16px", borderRadius: 10, border: "1px solid #d1d5db",
            background: "#fff", fontSize: 15, cursor: "pointer",
          }}
          title="Flip camera"
        >
          🔄
        </button>
        <button
          onClick={stopCamera}
          style={{
            padding: "12px 16px", borderRadius: 10, border: "1px solid #d1d5db",
            background: "#fff", fontSize: 15, cursor: "pointer",
          }}
        >
          ✕ Stop
        </button>
      </div>
    </div>
  );
}