"""
app.py — Sinhala OCR Flask API
Place this in: backend/app.py

Start the server:
    cd backend
    python app.py

Required folder layout:
    backend/
        app.py
        model_loader.py
        ocr_engine.py
        models/
            sinhala_cnn.h5
            label_map.json
            sinhala_yolo.pt    ← optional but recommended
"""

import base64
import logging
import os
import sys
import time

import numpy as np
from flask import Flask, jsonify, request
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")


# ─────────────────────────────────────────────
# NUMPY-SAFE JSON PROVIDER
# ─────────────────────────────────────────────
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, np.bool_):    return bool(obj)
        return super().default(obj)


# ─────────────────────────────────────────────
# FLASK APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Allow your frontend origins — add more if needed
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "*",   # remove this in production and keep only specific origins
]}})

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/x-bmp",
    "application/octet-stream",
}


# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
with app.app_context():
    from model_loader import get_model, get_label_map, models_ready, yolo_ready
    if models_ready():
        try:
            get_model()
            get_label_map()
            logger.info("✅ CNN + label map ready")
            if yolo_ready():
                logger.info("✅ YOLO weights found — segmentation mode: YOLO")
            else:
                logger.info("ℹ️  No YOLO weights — segmentation mode: connected-components")
        except Exception as exc:
            logger.error("❌ Failed to load models at startup: %s", exc)
    else:
        logger.warning(
            "⚠️  Model files are missing. "
            "Place sinhala_cnn.h5 and label_map.json inside backend/models/"
        )


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _is_allowed(filename: str, content_type: str) -> bool:
    ext = os.path.splitext(filename.lower())[1] if filename else ""
    return ext in ALLOWED_EXTENSIONS or content_type in ALLOWED_TYPES


def _run_ocr(image_bytes: bytes):
    """Call the OCR engine and return (result_dict, http_status_code)."""
    from ocr_engine import run_ocr
    start = time.time()
    try:
        result = run_ocr(image_bytes)
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        return result, 200
    except ValueError as exc:
        logger.error("ValueError in OCR: %s", exc)
        return {"error": str(exc)}, 422
    except MemoryError:
        return {"error": "Image too large to process."}, 413
    except Exception as exc:
        logger.exception("Unexpected OCR error: %s", exc)
        return {"error": f"OCR failed: {exc}"}, 500


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "service": "Sinhala OCR API",
        "version": "3.1",
        "status":  "running",
        "endpoints": {
            "GET  /health":          "Model + system status",
            "POST /predict":         "Upload image as multipart/form-data (field: 'file')",
            "POST /predict/base64":  "Send image as JSON {image_base64: '...'}",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    from model_loader import models_ready, get_label_map, yolo_ready
    ok  = models_ready()
    num = 0
    if ok:
        try:
            num = len(get_label_map())
        except Exception:
            ok = False

    return jsonify({
        "status":       "ok" if ok else "model_missing",
        "model_ready":  ok,
        "num_classes":  num,
        "yolo_ready":   yolo_ready(),
        "segmentation": "yolo" if yolo_ready() else "connected_components",
        "message":      (
            f"Ready — {num} Sinhala classes loaded."
            if ok else
            "Model files missing. See backend/models/"
        ),
    }), 200 if ok else 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept an image file via multipart/form-data.
    The field name must be 'file'.

    Example curl:
        curl -X POST http://localhost:5000/predict \
             -F "file=@handwriting.png"
    """
    if "file" not in request.files:
        return jsonify({
            "error": "No file provided. Send the image in a multipart field named 'file'."
        }), 400

    file         = request.files["file"]
    filename     = file.filename or ""
    content_type = file.content_type or ""

    logger.info("POST /predict  file='%s'  type='%s'", filename, content_type)

    if not filename:
        return jsonify({"error": "Uploaded file has no filename."}), 400

    if not _is_allowed(filename, content_type):
        return jsonify({
            "error": f"Unsupported file type '{content_type}'. Accepted: JPEG, PNG, WebP, BMP."
        }), 415

    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400

    logger.info("  → %d bytes received", len(image_bytes))
    result, status = _run_ocr(image_bytes)
    return jsonify(result), status


@app.route("/predict/base64", methods=["POST"])
def predict_base64():
    """
    Accept an image encoded as base64 JSON.

    Request body:
        { "image_base64": "data:image/png;base64,iVBORw0KGgo..." }
        or just the raw base64 string without the data URI prefix.

    Example curl:
        curl -X POST http://localhost:5000/predict/base64 \
             -H "Content-Type: application/json" \
             -d '{"image_base64": "<base64_string>"}'
    """
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({
            "error": "Request body must be JSON with field 'image_base64'."
        }), 400

    b64 = data["image_base64"]

    # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        return jsonify({"error": "Invalid base64 string."}), 400

    if not image_bytes:
        return jsonify({"error": "Decoded image is empty."}), 400

    logger.info("POST /predict/base64  size=%d bytes", len(image_bytes))
    result, status = _run_ocr(image_bytes)
    return jsonify(result), status


# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Image exceeds the 16 MB limit."}), 413


@app.errorhandler(404)
def not_found(_e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "HTTP method not allowed on this endpoint."}), 405


@app.errorhandler(500)
def internal_error(_e):
    return jsonify({"error": "Internal server error."}), 500


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("🚀 Starting Sinhala OCR API on http://0.0.0.0:%d", port)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)