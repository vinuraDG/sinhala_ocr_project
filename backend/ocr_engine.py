"""
ocr_engine.py
─────────────
Full OCR pipeline:
  raw image bytes
    → greyscale + denoise + binarize
    → deskew (fix tilted photos)
    → segment lines
    → segment characters per line
    → CNN prediction per character
    → combine into final Sinhala sentence

Key improvements over basic version:
  • Adaptive thresholding  — handles uneven lighting from phone photos
  • Deskewing              — corrects tilted handwriting
  • Morphological closing  — joins broken strokes in Sinhala characters
  • Connected-component analysis — more accurate than projection alone
  • Top-k predictions      — returns alternatives for low-confidence chars
  • Confidence filtering   — flags unreliable characters
  • Gap analysis           — detects word spaces accurately
"""

import cv2
import numpy as np
import logging
from collections import defaultdict

from model_loader import get_model, get_label_map

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────
IMAGE_SIZE      = (64, 64)   # must match training size
MIN_CHAR_WIDTH  = 8          # ignore blobs narrower than this (noise)
MIN_CHAR_HEIGHT = 10         # ignore blobs shorter than this (noise)
MIN_CHAR_AREA   = 80         # ignore blobs with fewer pixels (dots/specks)
CHAR_PADDING    = 6          # pixels of padding around each char crop
WORD_GAP_RATIO  = 1.8        # gap > avg_char_width * ratio → word space
TOP_K           = 3          # return top-3 predictions per character


# ═══════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESSING
# ═══════════════════════════════════════════════════════════════

def _decode_image(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes into a BGR OpenCV image."""
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(
            "Cannot decode image. "
            "Make sure you send a valid JPEG or PNG file."
        )
    return img


def _to_greyscale(bgr: np.ndarray) -> np.ndarray:
    """Convert BGR → greyscale."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _denoise(grey: np.ndarray) -> np.ndarray:
    """
    Remove camera noise while preserving stroke edges.
    Uses a fast Gaussian blur — sufficient for phone photos
    taken in reasonable lighting.
    """
    return cv2.GaussianBlur(grey, (3, 3), 0)


def _binarize(denoised: np.ndarray) -> np.ndarray:
    """
    Convert greyscale → binary (black/white) image.

    Strategy:
      1. Try adaptive thresholding first — handles uneven lighting,
         shadows from phone camera, paper texture.
      2. Fall back to Otsu global threshold if adaptive fails.

    Result: WHITE letters on BLACK background (inverted from paper).
    """
    # Adaptive threshold — best for phone photos with shadows
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,   # inverted: letter=white, bg=black
        blockSize=31,            # neighbourhood size (must be odd)
        C=10                     # constant subtracted from mean
    )

    # Also compute Otsu for comparison
    _, otsu = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Pick whichever gives more foreground (letter) pixels
    # Adaptive is usually better for uneven lighting
    adaptive_density = np.sum(adaptive > 0) / adaptive.size
    otsu_density     = np.sum(otsu > 0)     / otsu.size

    # Use adaptive unless it picks up WAY too much noise (>40% white)
    binary = adaptive if adaptive_density < 0.40 else otsu

    return binary


def _remove_noise(binary: np.ndarray) -> np.ndarray:
    """
    Morphological opening — removes tiny isolated white blobs (noise)
    without affecting actual character strokes.
    """
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)
    return cleaned


def _close_strokes(binary: np.ndarray) -> np.ndarray:
    """
    Morphological closing — joins small gaps within character strokes.
    Sinhala characters often have thin connecting lines that break when
    binarized. Closing fixes this before segmentation.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def _deskew(binary: np.ndarray) -> np.ndarray:
    """
    Correct skew (tilt) from phone photos of handwritten text.
    Uses the angle of the minimum-area bounding rectangle of
    all foreground pixels.

    Skips correction if the detected angle is very small (<0.5°)
    to avoid introducing resampling artifacts.
    """
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 50:
        return binary   # not enough content to estimate skew

    rect  = cv2.minAreaRect(coords)
    angle = rect[-1]   # degrees

    # minAreaRect returns angles in [-90, 0)
    # Convert to a skew angle in (-45, 45]
    if angle < -45:
        angle = 90 + angle

    # Skip tiny angles
    if abs(angle) < 0.5:
        return binary

    # Rotate image around its centre
    h, w = binary.shape
    centre  = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(centre, angle, 1.0)
    deskewed = cv2.warpAffine(
        binary, rot_mat, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    logger.debug("Deskewed by %.2f°", angle)
    return deskewed


def preprocess_image(image_bytes: bytes) -> tuple:
    """
    Full preprocessing pipeline.

    Returns
    -------
    binary   : np.ndarray  — clean binary image (white text, black bg)
    original : np.ndarray  — original BGR image (for annotation)
    grey     : np.ndarray  — greyscale image (for frontend display)
    steps    : dict        — intermediate images for debugging
    """
    original  = _decode_image(image_bytes)
    grey      = _to_greyscale(original)
    denoised  = _denoise(grey)
    binary    = _binarize(denoised)
    binary    = _remove_noise(binary)
    binary    = _close_strokes(binary)
    binary    = _deskew(binary)

    steps = {
        "original":  original,
        "grey":      grey,
        "denoised":  denoised,
        "binary":    binary,
    }
    return binary, original, grey, steps


# ═══════════════════════════════════════════════════════════════
# STEP 2 — LINE SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def segment_lines(binary: np.ndarray, v_padding: int = 6) -> list:
    """
    Detect text lines using horizontal projection profile.

    The projection sums pixel values row-by-row. Rows belonging
    to text have high sums; rows between lines have near-zero sums.

    Returns list of (y1, y2) tuples — one per detected line,
    sorted top-to-bottom.
    """
    h_proj    = np.sum(binary, axis=1).astype(np.float32)
    threshold = max(1.0, float(np.max(h_proj)) * 0.04)

    lines    = []
    in_line  = False
    start    = 0
    H        = binary.shape[0]

    for i, val in enumerate(h_proj):
        if val > threshold and not in_line:
            start   = i
            in_line = True
        elif val <= threshold and in_line:
            y1 = max(0, start - v_padding)
            y2 = min(H, i   + v_padding)
            # Skip lines that are too thin (likely noise)
            if (y2 - y1) >= MIN_CHAR_HEIGHT:
                lines.append((y1, y2))
            in_line = False

    if in_line:
        lines.append((max(0, start - v_padding), H))

    logger.debug("Lines found: %d", len(lines))
    return lines


# ═══════════════════════════════════════════════════════════════
# STEP 3 — CHARACTER SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def _connected_components(binary: np.ndarray, y1: int, y2: int) -> list:
    """
    Use connected-component analysis to find individual character blobs
    within a single line strip.

    This is more accurate than vertical projection alone because it
    handles overlapping or touching characters better.

    Returns list of dicts: {image, x1, x2, y1, y2, area}
    """
    line_strip = binary[y1:y2, :]
    H, W = line_strip.shape

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        line_strip, connectivity=8
    )

    chars = []
    for label in range(1, num_labels):   # 0 = background
        x      = stats[label, cv2.CC_STAT_LEFT]
        y      = stats[label, cv2.CC_STAT_TOP]
        w      = stats[label, cv2.CC_STAT_WIDTH]
        h      = stats[label, cv2.CC_STAT_HEIGHT]
        area   = stats[label, cv2.CC_STAT_AREA]

        # Filter noise blobs
        if w < MIN_CHAR_WIDTH or h < MIN_CHAR_HEIGHT or area < MIN_CHAR_AREA:
            continue

        # Add padding — crop from full binary not just line_strip
        x1 = max(0,   x      - CHAR_PADDING)
        x2 = min(W,   x + w  + CHAR_PADDING)
        cy1 = max(0,  y1 + y      - CHAR_PADDING)
        cy2 = min(binary.shape[0], y1 + y + h + CHAR_PADDING)

        char_img = binary[cy1:cy2, x1:x2]

        chars.append({
            "image": char_img,
            "x1":    x1,  "x2": x2,
            "y1":    cy1, "y2": cy2,
            "area":  area,
            "width": w,
        })

    # Sort left-to-right
    chars.sort(key=lambda c: c["x1"])
    return chars


def _merge_close_components(chars: list, merge_gap: int = 4) -> list:
    """
    Merge connected components that are very close horizontally.

    Sinhala vowel diacritics (matras) often appear as separate blobs
    very close to their base character. This step merges them back.
    """
    if not chars:
        return chars

    merged = [chars[0]]
    for ch in chars[1:]:
        prev = merged[-1]
        gap  = ch["x1"] - prev["x2"]

        if gap <= merge_gap:
            # Merge: extend bounding box, combine images
            new_x1 = min(prev["x1"], ch["x1"])
            new_x2 = max(prev["x2"], ch["x2"])
            new_y1 = min(prev["y1"], ch["y1"])
            new_y2 = max(prev["y2"], ch["y2"])

            # Re-crop from the stored binary region
            # (we store it in the char dict for this purpose)
            merged[-1] = {
                "x1":    new_x1, "x2": new_x2,
                "y1":    new_y1, "y2": new_y2,
                "area":  prev["area"] + ch["area"],
                "width": new_x2 - new_x1,
                # Image will be re-cropped in predict step
                "image": None,   # placeholder — set below
            }
        else:
            merged.append(ch)

    return merged


def segment_characters(binary: np.ndarray, lines: list) -> list:
    """
    Segment all characters across all lines.

    Returns flat list of character dicts:
    {
        image, x1, x2, y1, y2, line_idx, area, width
    }
    """
    all_chars = []

    for line_idx, (y1, y2) in enumerate(lines):
        raw_chars = _connected_components(binary, y1, y2)
        chars     = _merge_close_components(raw_chars, merge_gap=5)

        for ch in chars:
            # Re-crop image in case merge changed bounds
            if ch["image"] is None or ch["image"].size == 0:
                ch["image"] = binary[ch["y1"]:ch["y2"], ch["x1"]:ch["x2"]]

            if ch["image"].size == 0:
                continue

            ch["line_idx"] = line_idx
            all_chars.append(ch)

    logger.debug("Total characters segmented: %d", len(all_chars))
    return all_chars


# ═══════════════════════════════════════════════════════════════
# STEP 4 — CHARACTER PREDICTION
# ═══════════════════════════════════════════════════════════════

def _prepare_char_image(char_img: np.ndarray) -> np.ndarray:
    """
    Prepare a character image for CNN input:
      1. Pad to square (preserves aspect ratio — very important for
         Sinhala characters which have varying width/height)
      2. Resize to IMAGE_SIZE
      3. Normalize to [0.0, 1.0]
      4. Add batch + channel dimensions
    """
    h, w = char_img.shape[:2]

    # Pad to square
    size    = max(h, w)
    pad_top = (size - h) // 2
    pad_bot = size - h - pad_top
    pad_lft = (size - w) // 2
    pad_rgt = size - w - pad_lft

    padded = cv2.copyMakeBorder(
        char_img, pad_top, pad_bot, pad_lft, pad_rgt,
        cv2.BORDER_CONSTANT, value=0
    )

    # Resize
    resized     = cv2.resize(padded, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    normalized  = resized.astype("float32") / 255.0
    input_tensor = normalized.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    return input_tensor


def predict_character(char_img: np.ndarray, top_k: int = TOP_K) -> dict:
    """
    Run CNN on a single character image.

    Returns
    -------
    {
        letter      : str    — top predicted Sinhala letter
        confidence  : float  — confidence % (0–100)
        top_k       : list   — [{letter, confidence}, ...] for top-k
        low_conf    : bool   — True if confidence < 60%
    }
    """
    model     = get_model()
    label_map = get_label_map()

    input_tensor = _prepare_char_image(char_img)
    probs        = model.predict(input_tensor, verbose=0)[0]

    # Top-k indices sorted by probability (descending)
    top_indices = np.argsort(probs)[::-1][:top_k]

    best_idx    = int(top_indices[0])
    best_letter = label_map.get(best_idx, "?")
    best_conf   = round(float(probs[best_idx]) * 100, 2)

    top_k_results = [
        {
            "letter":     label_map.get(int(i), "?"),
            "confidence": round(float(probs[i]) * 100, 2),
        }
        for i in top_indices
    ]

    return {
        "letter":     best_letter,
        "confidence": best_conf,
        "top_k":      top_k_results,
        "low_conf":   best_conf < 60.0,
    }


def predict_all_characters(all_chars: list) -> list:
    """
    Predict every character in the segmented list.
    Adds prediction fields to each character dict.
    Returns updated list.
    """
    results = []
    for ch in all_chars:
        pred = predict_character(ch["image"])
        results.append({
            **ch,
            "letter":     pred["letter"],
            "confidence": pred["confidence"],
            "top_k":      pred["top_k"],
            "low_conf":   pred["low_conf"],
        })
        logger.debug(
            "Char at x=%d → '%s'  (%.1f%%)",
            ch["x1"], pred["letter"], pred["confidence"]
        )
    return results


# ═══════════════════════════════════════════════════════════════
# STEP 5 — SENTENCE ASSEMBLY
# ═══════════════════════════════════════════════════════════════

def _detect_word_gaps(line_chars: list) -> list:
    """
    Detect word boundaries within a line by analysing horizontal gaps.

    Algorithm:
      1. Compute gap between each consecutive pair of characters.
      2. Compute average character width for this line.
      3. If gap > avg_char_width * WORD_GAP_RATIO → word space.

    Returns list of (char_dict, insert_space_before) tuples.
    """
    if not line_chars:
        return []

    # Average character width for this line
    avg_w = np.mean([ch["width"] for ch in line_chars]) if line_chars else 20

    result = [(line_chars[0], False)]   # first char never has a space before it

    for i in range(1, len(line_chars)):
        prev = line_chars[i - 1]
        curr = line_chars[i]
        gap  = curr["x1"] - prev["x2"]
        insert_space = gap > (avg_w * WORD_GAP_RATIO)
        result.append((curr, insert_space))

    return result


def build_sentence(char_results: list) -> dict:
    """
    Assemble predicted characters into a full Sinhala sentence.

    Returns
    -------
    {
        text          : str   — full sentence (lines joined with \\n)
        lines         : list  — per-line text strings
        line_details  : list  — per-line list of character dicts
        avg_confidence: float — overall average confidence %
    }
    """
    # Group by line index
    lines_dict = defaultdict(list)
    for ch in char_results:
        lines_dict[ch["line_idx"]].append(ch)

    sentence_lines  = []
    line_details    = []
    all_confidences = []

    for line_idx in sorted(lines_dict.keys()):
        # Sort characters left-to-right within each line
        line_chars = sorted(lines_dict[line_idx], key=lambda c: c["x1"])

        # Detect word gaps
        spaced = _detect_word_gaps(line_chars)

        line_text = ""
        for ch, insert_space in spaced:
            if insert_space:
                line_text += " "
            line_text += ch["letter"]
            all_confidences.append(ch["confidence"])

        sentence_lines.append(line_text)
        line_details.append(line_chars)

    full_text      = "\n".join(sentence_lines)
    avg_confidence = round(float(np.mean(all_confidences)), 2) \
                     if all_confidences else 0.0

    return {
        "text":           full_text,
        "lines":          sentence_lines,
        "line_details":   line_details,
        "avg_confidence": avg_confidence,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════

def run_ocr(image_bytes: bytes) -> dict:
    """
    Full OCR pipeline: raw image bytes → Sinhala text.

    Returns
    -------
    {
        text          : str    — recognized full sentence
        lines         : list   — per-line text
        characters    : list   — per-character details
        line_count    : int
        char_count    : int
        avg_confidence: float
        preprocessing : dict   — info about preprocessing steps
    }
    """
    # ── 1. Preprocess ─────────────────────────────────────────
    binary, original, grey, steps = preprocess_image(image_bytes)
    h, w = binary.shape
    logger.info("Image preprocessed: %dx%d", w, h)

    # ── 2. Segment lines ──────────────────────────────────────
    lines = segment_lines(binary)
    if not lines:
        return {
            "text":           "",
            "lines":          [],
            "characters":     [],
            "line_count":     0,
            "char_count":     0,
            "avg_confidence": 0.0,
            "error":          "No text lines detected. "
                              "Try a clearer photo with better lighting.",
        }
    logger.info("Lines detected: %d", len(lines))

    # ── 3. Segment characters ─────────────────────────────────
    all_chars = segment_characters(binary, lines)
    if not all_chars:
        return {
            "text":           "",
            "lines":          [],
            "characters":     [],
            "line_count":     len(lines),
            "char_count":     0,
            "avg_confidence": 0.0,
            "error":          "No characters detected within lines. "
                              "Check image quality and lighting.",
        }
    logger.info("Characters detected: %d", len(all_chars))

    # ── 4. Predict each character ─────────────────────────────
    char_results = predict_all_characters(all_chars)

    # ── 5. Build sentence ─────────────────────────────────────
    sentence_data = build_sentence(char_results)

    # ── 6. Build clean response ───────────────────────────────
    serializable_chars = []
    for ch in char_results:
        serializable_chars.append({
            "letter":     ch["letter"],
            "confidence": ch["confidence"],
            "low_conf":   ch["low_conf"],
            "top_k":      ch["top_k"],
            "x1": ch["x1"], "x2": ch["x2"],
            "y1": ch["y1"], "y2": ch["y2"],
            "line_idx":   ch["line_idx"],
        })

    return {
        "text":           sentence_data["text"],
        "lines":          sentence_data["lines"],
        "characters":     serializable_chars,
        "line_count":     len(lines),
        "char_count":     len(char_results),
        "avg_confidence": sentence_data["avg_confidence"],
        "image_size":     {"width": w, "height": h},
    }
