"""
model_loader.py
───────────────
Loads the trained Sinhala CNN model and label map ONCE at server
startup and reuses them for every prediction request.
This avoids reloading the model on every API call (which is slow).
"""

import json
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths (relative to backend/) ─────────────────────────────
MODEL_PATH    = os.path.join(os.path.dirname(__file__), "models", "sinhala_cnn.h5")
LABELMAP_PATH = os.path.join(os.path.dirname(__file__), "models", "label_map.json")

# ── Module-level singletons ───────────────────────────────────
_model     = None
_label_map = None   # int → Sinhala letter


# ─────────────────────────────────────────────────────────────
def get_model():
    """
    Load the Keras CNN model once and cache it.
    Raises FileNotFoundError if the .h5 file is missing.
    """
    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at: {MODEL_PATH}\n"
            "Train the model in Kaggle Notebook 02 and download "
            "sinhala_cnn.h5 into backend/models/"
        )

    logger.info("⏳ Loading CNN model from %s ...", MODEL_PATH)

    # Import here so the app can start even without TF installed
    # (useful for running tests / health-check only mode)
    from tensorflow.keras.models import load_model
    _model = load_model(MODEL_PATH)

    logger.info(
        "✅ Model loaded  |  input shape: %s  |  classes: %d",
        _model.input_shape,
        _model.output_shape[-1],
    )
    return _model


# ─────────────────────────────────────────────────────────────
def get_label_map():
    """
    Load label_map.json once and cache it.
    Returns dict: int → Sinhala letter string.
    Raises FileNotFoundError if the JSON file is missing.
    """
    global _label_map

    if _label_map is not None:
        return _label_map

    if not os.path.exists(LABELMAP_PATH):
        raise FileNotFoundError(
            f"Label map not found at: {LABELMAP_PATH}\n"
            "Download label_map.json from Kaggle Notebook 02 output "
            "into backend/models/"
        )

    with open(LABELMAP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # JSON keys are always strings — convert to int
    _label_map = {int(k): v for k, v in raw.items()}

    logger.info("✅ Label map loaded  |  %d classes", len(_label_map))
    return _label_map


# ─────────────────────────────────────────────────────────────
def get_num_classes():
    """Return the number of Sinhala character classes."""
    return len(get_label_map())


# ─────────────────────────────────────────────────────────────
def models_ready():
    """
    Check whether both model files exist on disk.
    Used by the /health endpoint.
    """
    return os.path.exists(MODEL_PATH) and os.path.exists(LABELMAP_PATH)
