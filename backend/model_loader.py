"""
model_loader.py — lazy singleton loader for CNN + label map + YOLO
Place this in: backend/model_loader.py

Models are loaded once on first use (lazy loading).
Thread-safe for Flask's threaded mode.
"""

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "sinhala_cnn.h5")
LABELMAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")
YOLO_PATH     = os.path.join(BASE_DIR, "models", "sinhala_yolo.pt")

_model      = None
_label_map  = None
_yolo       = None
_lock       = threading.Lock()


# ─────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────
def get_model():
    global _model
    if _model is not None:
        return _model

    with _lock:
        if _model is not None:        # double-checked locking
            return _model

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"CNN model not found: {MODEL_PATH}\n"
                f"Place sinhala_cnn.h5 inside backend/models/"
            )

        try:
            from tensorflow.keras.models import load_model
        except ImportError:
            raise ImportError(
                "TensorFlow is not installed. Run:\n"
                "  pip install tensorflow"
            )

        try:
            _model = load_model(MODEL_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load CNN model: {e}")

        try:
            logger.info(
                "CNN loaded — input: %s  output classes: %d",
                _model.input_shape,
                _model.output_shape[-1],
            )
        except Exception:
            logger.info("CNN loaded successfully.")

    return _model


# ─────────────────────────────────────────────
# LABEL MAP
# ─────────────────────────────────────────────
def get_label_map():
    global _label_map
    if _label_map is not None:
        return _label_map

    with _lock:
        if _label_map is not None:
            return _label_map

        if not os.path.exists(LABELMAP_PATH):
            raise FileNotFoundError(
                f"Label map not found: {LABELMAP_PATH}\n"
                f"Place label_map.json inside backend/models/"
            )

        if os.path.getsize(LABELMAP_PATH) == 0:
            raise ValueError("label_map.json is empty.")

        try:
            with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in label_map.json: {e}")

        if not isinstance(raw, dict):
            raise ValueError("label_map.json must be a JSON object mapping index → character.")

        try:
            _label_map = {int(k): str(v) for k, v in raw.items()}
        except Exception as e:
            raise ValueError(f"Error parsing label_map keys: {e}")

        logger.info("Label map loaded — %d classes.", len(_label_map))

    return _label_map


# ─────────────────────────────────────────────
# YOLO MODEL (optional)
# ─────────────────────────────────────────────
def get_yolo():
    global _yolo
    if _yolo is not None:
        return _yolo

    if not os.path.exists(YOLO_PATH):
        return None     # YOLO weights absent — caller falls back to CC

    with _lock:
        if _yolo is not None:
            return _yolo

        try:
            from ultralytics import YOLO
            _yolo = YOLO(YOLO_PATH)
            logger.info("YOLO loaded — %s", YOLO_PATH)
            return _yolo
        except ImportError:
            logger.warning(
                "ultralytics not installed — YOLO disabled. "
                "Run: pip install ultralytics"
            )
            return None
        except Exception as e:
            logger.warning("Failed to load YOLO model (%s) — falling back to CC.", e)
            return None


def yolo_ready() -> bool:
    """Returns True if the YOLO weights file exists on disk."""
    return os.path.exists(YOLO_PATH)


# ─────────────────────────────────────────────
# STATUS CHECK
# ─────────────────────────────────────────────
def models_ready() -> bool:
    """Check that mandatory model files (CNN + label map) exist."""
    cnn_ok   = os.path.exists(MODEL_PATH)
    label_ok = os.path.exists(LABELMAP_PATH)

    if not cnn_ok:
        logger.error("Missing CNN model: %s", MODEL_PATH)
    if not label_ok:
        logger.error("Missing label map: %s", LABELMAP_PATH)

    if yolo_ready():
        logger.info("YOLO weights found — character segmentation: YOLO")
    else:
        logger.info("No YOLO weights — character segmentation: connected-components")

    return cnn_ok and label_ok