import base64
import logging
import os
import sys
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("app")

app = Flask(__name__)

# Allow React dev server (port 3000) and Vite dev server (port 5173)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]}})

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024


with app.app_context():
    from model_loader import get_model, get_label_map, models_ready
    if models_ready():
        try:
            get_model()
            get_label_map()
            logger.info("✅ Model and label map loaded — ready!")
        except Exception as e:
            logger.error("❌ Failed to load model: %s", e)
    else:
        logger.warning("⚠️  Model files missing — place sinhala_cnn.h5 and label_map.json in backend/models/")


def _run_ocr_safe(image_bytes):
    from ocr_engine import run_ocr
    start = time.time()
    try:
        result = run_ocr(image_bytes)
        result["processing_time_ms"] = round((time.time() - start) * 1000)
        return result, 200
    except ValueError as e:
        return {"error": str(e)}, 422
    except MemoryError:
        return {"error": "Image too large to process."}, 413
    except Exception as e:
        logger.exception("OCR error")
        return {"error": f"OCR processing failed: {str(e)}"}, 500


@app.route("/", methods=["GET"])
def root():
    return jsonify({"service": "Sinhala OCR API", "version": "2.0", "status": "running"})


@app.route("/health", methods=["GET"])
def health():
    from model_loader import models_ready, get_label_map
    model_ok    = models_ready()
    num_classes = 0
    if model_ok:
        try:
            num_classes = len(get_label_map())
        except Exception:
            model_ok = False
    code = 200 if model_ok else 503
    return jsonify({
        "status":      "ok" if model_ok else "model_missing",
        "model_ready": model_ok,
        "num_classes": num_classes,
        "message":     f"Ready — {num_classes} classes." if model_ok else "Model files missing.",
    }), code


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file in request. Field name must be 'file'."}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "File has no name."}), 400
    if file.content_type not in ALLOWED_TYPES:
        return jsonify({"error": f"Unsupported type: {file.content_type}. Use JPEG/PNG/WebP/BMP."}), 415
    image_bytes = file.read()
    if not image_bytes:
        return jsonify({"error": "Uploaded file is empty."}), 400
    logger.info("POST /predict  file=%s  size=%d bytes", file.filename, len(image_bytes))
    result, status = _run_ocr_safe(image_bytes)
    return jsonify(result), status


@app.route("/predict/base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "Body must be JSON with field 'image_base64'."}), 400
    b64 = data["image_base64"]
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(b64)
    except Exception:
        return jsonify({"error": "Invalid base64 string."}), 400
    if not image_bytes:
        return jsonify({"error": "Decoded image is empty."}), 400
    logger.info("POST /predict/base64  size=%d bytes", len(image_bytes))
    result, status = _run_ocr_safe(image_bytes)
    return jsonify(result), status


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Image too large. Max 16 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    logger.info("🚀 Starting on port %d", port)
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)