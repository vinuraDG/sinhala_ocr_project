import json
import logging
import os

logger = logging.getLogger(__name__)

MODEL_PATH    = os.path.join(os.path.dirname(__file__), "models", "sinhala_cnn.h5")
LABELMAP_PATH = os.path.join(os.path.dirname(__file__), "models", "label_map.json")

_model     = None
_label_map = None


def get_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    from tensorflow.keras.models import load_model
    _model = load_model(MODEL_PATH)
    logger.info("Model loaded — input: %s, classes: %d", _model.input_shape, _model.output_shape[-1])
    return _model


def get_label_map():
    global _label_map
    if _label_map is not None:
        return _label_map
    if not os.path.exists(LABELMAP_PATH):
        raise FileNotFoundError(f"Label map not found: {LABELMAP_PATH}")
    with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    _label_map = {int(k): v for k, v in raw.items()}
    logger.info("Label map loaded — %d classes", len(_label_map))
    return _label_map


def models_ready():
    return os.path.exists(MODEL_PATH) and os.path.exists(LABELMAP_PATH)