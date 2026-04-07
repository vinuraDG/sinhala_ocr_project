import React, { useState, useRef, useCallback } from "react";
import "./CameraCapture.css";

export default function CameraCapture({ onCapture }) {
  const [stream, setStream] = useState(null);
  const [facingMode, setFacingMode] = useState("environment");
  const [error, setError] = useState(null);

  const videoRef = useRef();
  const canvasRef = useRef();

  const startCamera = useCallback(async (facing = facingMode) => {
    setError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing },
        audio: false,
      });
      setStream(s);
      // Small delay so the video element is mounted
      setTimeout(() => {
        if (videoRef.current) videoRef.current.srcObject = s;
      }, 100);
    } catch {
      setError("Camera access denied. Please allow camera permission and try again.");
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
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);

    const dataURL = canvas.toDataURL("image/jpeg", 0.92);
    canvas.toBlob((blob) => {
      const file = new File([blob], "captured.jpg", { type: "image/jpeg" });
      onCapture({ dataURL, file });
    }, "image/jpeg", 0.92);

    stopCamera();
  };

  return (
    <div className="camera-capture">
      {stream ? (
        <div className="camera-live-box">
          <div className="camera-viewfinder">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="camera-stream"
            />
            <div className="viewfinder-corners">
              <span /><span /><span /><span />
            </div>
            <div className="camera-hint">Align handwritten text within frame</div>
          </div>
          <canvas ref={canvasRef} className="camera-canvas" />
          <div className="camera-controls">
            <button className="snap-btn" onClick={snapPhoto}>
              📸 Capture
            </button>
            <button className="flip-btn" onClick={flipCamera} title="Flip camera">⟳</button>
            <button className="stop-btn" onClick={stopCamera}>✕ Stop</button>
          </div>
          <div className="camera-tips">
            <div className="tip-row">Ensure good lighting and flat paper</div>
            <div className="tip-row">Hold steady before capturing</div>
          </div>
        </div>
      ) : (
        <div className="cam-placeholder">
          <div className="cam-icon">📷</div>
          <p className="cam-title">Camera is off</p>
          <p className="cam-sub">
            Click below to activate your camera and capture handwritten Sinhala text.
          </p>
          {error && <div className="error-msg">{error}</div>}
          <button className="start-camera-btn" onClick={() => startCamera()}>
            📷 Start Camera
          </button>
        </div>
      )}
      {error && stream && <div className="error-msg" style={{ marginTop: 12 }}>{error}</div>}
    </div>
  );
}
