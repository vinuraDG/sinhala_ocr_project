import React from "react";
import { Link } from "react-router-dom";
import "./Footer.css";

export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div className="footer-brand">
          <div className="footer-logo">
            <span className="brand-icon-sm">අ</span>
            <span className="footer-brand-name">Akshara Scan</span>
          </div>
          <p className="footer-tagline">
            Digitizing Sinhala handwriting with deep learning.
          </p>
        </div>

        <div className="footer-links-group">
          <div className="footer-col">
            <span className="footer-col-title">Navigate</span>
            <Link to="/" className="footer-link">Home</Link>
            <Link to="/upload" className="footer-link">Upload Image</Link>
            <Link to="/camera" className="footer-link">Camera Capture</Link>
          </div>
          <div className="footer-col">
            <span className="footer-col-title">Project</span>
            <span className="footer-link muted">Final Year</span>
            <span className="footer-link muted">University of Greenwich</span>
            <span className="footer-link muted">BSc (Hons) Computing</span>
          </div>
        </div>
      </div>

      <div className="footer-bottom">
        <span>© {new Date().getFullYear()} A. G. Pooja Sathsarani Bandara</span>
      </div>
    </footer>
  );
}
