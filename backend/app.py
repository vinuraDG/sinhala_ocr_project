"""
app.py
──────
Flask REST API for Sinhala Handwritten OCR.

Endpoints:
  GET  /health          — server + model status check
  POST /predict         — main OCR endpoint (image → text)
  POST /predict/base64  — same but accepts base64-encoded image

Run:
  python app.py
  → http://localhost:5000
"""

import base64
import logging
import os
import sys
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

# ── Flask app ─────────────────────────────────────────────────
app = Flask(__name__)

# Allow React dev server and production origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",   # React dev
    "http://localhost:5173",   # Vite dev
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

CORS(app, resources={r"/*": {"origins": ALLOWED_ORIGINS}})

# ── Allowed MIME types ────────────────────────────────────────
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg",
    "image/png",  "image/webp",
    "image/bmp",
}

# ── Max upload size: 16 MB ────────────────────────────────────
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


# ═══════════════════════════════════════════════════════════════
# Pre-load model at startup
# ═══════════════════════════════════════════════════════════════

def load_models_on_startup():
    """
    Pre-load CNN model and label map when Flask starts.
    Crashes with a clear message if files are missing.
    """
    from model_loader import get_model, get_label_map, models_ready

    if not models_ready():
        logger.error(
            "❌  Model files missing in backend/models/\n"
            "    Required files:\n"
            "      • models/sinhala_cnn.h5\n"
            "      • models/label_map.json\n"
            "    Train in Kaggle Notebook 02 then download both files."
        )
        # Continue running so /health endpoint works and shows clear error
        return False

    try:
        get_model()
        get_label_map()
        logger.info("✅ Model and label map loaded — ready to serve!")
        return True
    except Exception as e:
        logger.error("❌ Failed to load model: %s", e)
        return False


with app.app_context():
    _models_ok = load_models_on_startup()


# ═══════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════

def _run_ocr_safe(image_bytes: bytes) -> tuple:
    """
    Run OCR and return (result_dict, http_status_code).
    Wraps errors in consistent JSON structure.
    """
    from ocr_engine import run_ocr

    start = time.time()
    try:
        result = run_ocr(image_bytes)
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        return result, 200

    except ValueError as e:
        logger.warning("Value error during OCR: %s", e)
        return {"error": str(e)}, 422

    except MemoryError:
        logger.error("Out of memory during OCR")
        return {"error": "Image too large to process. Please send a smaller image."}, 413

    except Exception as e:
        logger.exception("Unexpected OCR error")
        return {"error": f"OCR processing failed: {str(e)}"}, 500


# ═══════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Sinhala Handwritten OCR API",
        "version": "2.0",
        "status":  "running",
        "endpoints": {
            "health":         "GET  /health",
            "predict":        "POST /predict         (multipart/form-data, field: file)",
            "predict_base64": "POST /predict/base64  (JSON: {image_base64: '...'})",
        },
    })


# ─────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """
    Health check — used by frontend to verify backend is ready.
    Returns model status and basic system info.
    """
    from model_loader import models_ready, get_label_map

    model_ok = models_ready()
    num_classes = 0
    if model_ok:
        try:
            num_classes = len(get_label_map())
        except Exception:
            model_ok = False

    status_code = 200 if model_ok else 503
    return jsonify({
        "status":      "ok" if model_ok else "model_missing",
        "model_ready": model_ok,
        "num_classes": num_classes,
        "message":     (
            f"Ready — {num_classes} Sinhala character classes loaded."
            if model_ok
            else "Model files missing. See backend/models/ directory."
        ),
    }), status_code


# ─────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    """
    Main OCR endpoint.

    Request:  multipart/form-data
              field "file" = image (JPEG / PNG / WebP / BMP)

    Response JSON:
    {
        "text":           "සිංහල වාක්‍යය",
        "lines":          ["line1", "line2"],
        "characters":     [
            {
                "letter":     "ස",
                "confidence": 94.2,
                "low_conf":   false,
                "top_k":      [{"letter":"ස","confidence":94.2}, ...],
                "x1": 10, "x2": 42, "y1": 5, "y2": 60,
                "line_idx": 0
            },
            ...
        ],
        "line_count":        1,
        "char_count":        8,
        "avg_confidence":    87.3,
        "processing_time_ms": 142
    }
    """
    # ── Validate file ──────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({
            "error": "No file in request. "
                     "Send image as multipart/form-data with field name 'file'."
        }), 400

    file = request.files["file"]

    if file.filename == "" or file.filename is None:
        return jsonify({"error": "File has no name. Please attach a valid image."}), 400

    if file.content_type not in ALLOWED_TYPES:
        return jsonify({
            "error": f"Unsupported file type: '{file.content_type}'. "
                     f"Accepted: JPEG, PNG, WebP, BMP."
        }), 415

    # ── Read and process ───────────────────────────────────────
    image_bytes = file.read()
    if len(image_bytes) == 0:
        return jsonify({"error": "Uploaded file is empty."}), 400

    logger.info(
        "POST /predict  |  file='%s'  |  size=%d bytes",
        file.filename, len(image_bytes)
    )

    result, status = _run_ocr_safe(image_bytes)
    return jsonify(result), status


# ─────────────────────────────────────────────────────────────
@app.route("/predict/base64", methods=["POST"])
def predict_base64():
    """
    Alternative OCR endpoint for base64-encoded images.
    Useful when sending photos from mobile browsers (camera capture).

    Request JSON:
    {
        "image_base64": "<base64 string>",
        "mime_type":    "image/jpeg"   (optional, default: image/jpeg)
    }
    """
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({
            "error": "Request body must be JSON with field 'image_base64'."
        }), 400

    b64_string = data["image_base64"]

    # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64_string)
    except Exception:
        return jsonify({"error": "Invalid base64 string."}), 400

    if len(image_bytes) == 0:
        return jsonify({"error": "Decoded image is empty."}), 400

    logger.info(
        "POST /predict/base64  |  size=%d bytes", len(image_bytes)
    )

    result, status = _run_ocr_safe(image_bytes)
    return jsonify(result), status


# ═══════════════════════════════════════════════════════════════
# Error handlers
# ═══════════════════════════════════════════════════════════════

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        "error": "Image file too large. Maximum size is 16 MB."
    }), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found.",
        "available": ["GET /health", "POST /predict", "POST /predict/base64"],
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


# ═══════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    logger.info("🚀 Starting Sinhala OCR API on port %d", port)
    logger.info("   Debug mode : %s", debug)
    logger.info("   CORS allow : %s", ALLOWED_ORIGINS)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True,   # handle multiple requests simultaneously
    )
