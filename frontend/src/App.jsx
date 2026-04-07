import React, { useState, useRef, useCallback } from "react";
import { BrowserRouter, Routes, Route, Link, useLocation } from "react-router-dom";
import CameraCapture from "./components/CameraCapture";
import ResultDisplay from "./components/ResultDisplay";
import Footer from "./components/Footer";
import { recognizeText} from "./api/ocrApi";
import "./index.css";
import "./App.css";

/* ─── Navbar ─────────────────────────────────────────────── */
function Navbar() {
  const { pathname } = useLocation();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  React.useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", fn);
    return () => window.removeEventListener("scroll", fn);
  }, []);

  React.useEffect(() => setOpen(false), [pathname]);

  const links = [
    { to: "/", label: "Home" },
    { to: "/upload", label: "Upload" },
    { to: "/camera", label: "Camera" },
  ];

  return (
    <nav className={`navbar ${scrolled ? "scrolled" : ""}`}>
      <Link to="/" className="nav-brand">
        <span className="brand-icon">අ</span>
        <span className="brand-text">Akshara Scan</span>
      </Link>
      <div className={`nav-links ${open ? "open" : ""}`}>
        {links.map((l) => (
          <Link key={l.to} to={l.to} className={`nav-link ${pathname === l.to ? "active" : ""}`}>
            {l.label}
          </Link>
        ))}
        <Link to="/upload" className="nav-cta">Try Now</Link>
      </div>
      <button className={`hamburger ${open ? "open" : ""}`} onClick={() => setOpen(!open)}>
        <span /><span /><span />
      </button>
    </nav>
  );
}

/* ─── Home page ───────────────────────────────────────────── */
function Home() {
  const features = [
    { icon: "📷", title: "Camera Capture", desc: "Snap a photo directly from your phone or webcam for instant recognition." },
    { icon: "📤", title: "Image Upload", desc: "Upload an existing image of handwritten Sinhala text from your device." },
    { icon: "⚡", title: "Instant Results", desc: "Our CNN model identifies each character and returns Unicode text in milliseconds." },
    { icon: "📊", title: "Confidence Scores", desc: "Get per-character confidence scores to understand recognition accuracy." },
  ];
  const steps = [
    { num: "01", label: "Capture or Upload", desc: "Use your camera or upload a handwritten Sinhala image." },
    { num: "02", label: "AI Processing", desc: "Our CNN model segments and recognizes each character." },
    { num: "03", label: "Get Digital Text", desc: "Receive editable Unicode text ready to copy and use." },
  ];

  return (
    <div className="home">
      <section className="hero">
        <div className="hero-eyebrow"><span className="eyebrow-dot" />Sinhala</div>
        <h1 className="hero-title">
          Handwritten Sinhala,<br />
          <span className="title-accent">Digitized Instantly</span>
        </h1>
        <p className="hero-sub">
          A deep learning system that converts handwritten Sinhala script into editable Unicode text.
          Powered by a CNN trained on hundreds of character samples.
        </p>
        <div className="hero-actions">
          <Link to="/upload" className="btn-primary">Upload an Image</Link>
          <Link to="/camera" className="btn-secondary">Open Camera</Link>
        </div>
        <div className="hero-sinhala-deco" aria-hidden="true">
          <span>ශ්‍රී</span><span>ල</span><span>ං</span><span>කා</span>
        </div>
      </section>

      <section className="steps-section">
        <div className="section-label">How it works</div>
        <div className="steps-row">
          {steps.map((s, i) => (
            <div className="step-card" key={i} style={{ animationDelay: `${i * 100}ms` }}>
              <div className="step-num">{s.num}</div>
              <h3 className="step-title">{s.label}</h3>
              <p className="step-desc">{s.desc}</p>
              {i < steps.length - 1 && <div className="step-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      <section className="features-section">
        <div className="section-label">Features</div>
        <h2 className="section-title">Everything you need</h2>
        <div className="features-grid">
          {features.map((f, i) => (
            <div className="feature-card" key={i} style={{ animationDelay: `${i * 80}ms` }}>
              <span className="feature-icon">{f.icon}</span>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="cta-section">
        <div className="cta-box">
          <div className="cta-glyph" aria-hidden="true">අ</div>
          <h2 className="cta-title">Ready to digitize?</h2>
          <p className="cta-sub">Start converting handwritten Sinhala text to digital format right now.</p>
          <Link to="/upload" className="btn-primary">Get Started →</Link>
        </div>
      </section>
    </div>
  );
}

/* ─── Shared recognize logic ──────────────────────────────── */
function useRecognize() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const recognize = async (file) => {
    setLoading(true);
    setError(null);
    try {
      let res;
      res = await recognizeText(file);
      setResult(res);
    } catch {
      setError("Recognition failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const reset = () => { setResult(null); setError(null); };

  return { result, loading, error, setError, recognize, reset };
}

/* ─── Upload page ─────────────────────────────────────────── */
function UploadPage() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileRef = useRef();
  const { result, loading, error, setError, recognize, reset } = useRecognize();

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) {
      setError("Please upload a valid image file (JPG, PNG, etc.)");
      return;
    }
    setFile(f);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(f);
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  }, []);

  const handleReset = () => { setFile(null); setPreview(null); reset(); };

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <div className="page-tag">Image Upload</div>
        <h1 className="page-title">Upload Handwritten Image</h1>
        <p className="page-sub">Upload a photo of handwritten Sinhala text and our AI will convert it to editable Unicode.</p>
      </div>

      {result ? (
        <ResultDisplay result={result} onReset={handleReset} />
      ) : !preview ? (
        <>
          <div
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onClick={() => fileRef.current.click()}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
          >
            <div className="drop-icon">📄</div>
            <p className="drop-title">Drop your image here</p>
            <p className="drop-sub">or click to browse files</p>
            <span className="drop-formats">JPG · PNG · WEBP · BMP</span>
            <input ref={fileRef} type="file" accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])} className="hidden-input" />
          </div>
          {error && <div className="error-msg mt-12">{error}</div>}
        </>
      ) : (
        <div className="preview-area">
          <div className="preview-header">
            <span className="preview-label">Selected Image</span>
            <button className="change-btn" onClick={handleReset}>Change</button>
          </div>
          <img src={preview} alt="Preview" className="preview-img" />
          <div className="preview-meta">
            <span>{file.name}</span>
            <span>{(file.size / 1024).toFixed(1)} KB</span>
          </div>
          {error && <div className="error-msg">{error}</div>}
          <button
            className={`recognize-btn ${loading ? "loading" : ""}`}
            onClick={() => recognize(file)}
            disabled={loading}
          >
            {loading ? <><span className="spinner" />Recognizing...</> : "Recognize Text →"}
          </button>
        </div>
      )}
    </div>
  );
}

/* ─── Camera page ─────────────────────────────────────────── */
function CameraPage() {
  const [captured, setCaptured] = useState(null);
  const { result, loading, error, recognize, reset } = useRecognize();

  const handleCapture = ({ dataURL, file }) => {
    setCaptured({ dataURL, file });
  };

  const handleReset = () => { setCaptured(null); reset(); };

  return (
    <div className="page-wrapper">
      <div className="page-header">
        <div className="page-tag">Live Camera</div>
        <h1 className="page-title">Capture Handwriting</h1>
        <p className="page-sub">Point your camera at handwritten Sinhala text and snap a photo for instant recognition.</p>
      </div>

      {result ? (
        <ResultDisplay result={result} onReset={handleReset} />
      ) : captured ? (
        <div className="preview-area">
          <div className="preview-header">
            <span className="preview-label">Captured Photo</span>
            <button className="change-btn" onClick={handleReset}>Retake</button>
          </div>
          <img src={captured.dataURL} alt="Captured" className="preview-img" />
          {error && <div className="error-msg">{error}</div>}
          <button
            className={`recognize-btn ${loading ? "loading" : ""}`}
            onClick={() => recognize(captured.file)}
            disabled={loading}
            style={{ marginTop: 20 }}
          >
            {loading ? <><span className="spinner" />Recognizing...</> : "Recognize Text →"}
          </button>
        </div>
      ) : (
        <CameraCapture onCapture={handleCapture} />
      )}
    </div>
  );
}

/* ─── 404 Not Found ───────────────────────────────────────── */
function NotFound() {
  return (
    <div className="page-wrapper" style={{ textAlign: "center", paddingTop: "160px" }}>
      <div style={{ fontSize: "4rem", marginBottom: "16px" }}>404</div>
      <h1 className="page-title">Page Not Found</h1>
      <p className="page-sub" style={{ margin: "12px auto 32px" }}>
        The page you're looking for doesn't exist.
      </p>
      <Link to="/" className="btn-primary">← Back to Home</Link>
    </div>
  );
}

/* ─── Root App ────────────────────────────────────────────── */
export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/camera" element={<CameraPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}
