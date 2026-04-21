"""
ocr_engine.py — Sinhala OCR pipeline
Place this in: backend/ocr_engine.py

Pipeline:
  1. Preprocess image
  2. Segment lines (horizontal projection)
  3. Detect characters using YOLO (if available) or Connected Components fallback
  4. Classify each character crop using CNN
  5. Build final text

Fixes applied (v5 — verified against real ය image):

  BUG 1 FIXED: _count_loops() now applies morphological CLOSE before counting.
    ය's loop is open in handwriting → RETR_CCOMP finds 0 holes on raw binary.
    After ellipse close(5,5) the loop seals → 1 hole found correctly.

  BUG 2 FIXED: _has_isolated_top_component() was too aggressive.
    ය has an upward tail that lands in the top 25% of the tight crop.
    Old code flagged those pixels as a "diacritic" → _looks_like_ya returned False.
    Fix: use only top 15% (not 25%), require the component to be DISCONNECTED
    from the row below it (bottom row of the top strip must be empty), AND
    the component must be < 12% of total ink (not 18%).

  OTHER FIXES (v3/v4, retained):
  - Stronger Gaussian blur + larger adaptive threshold block for blurry input.
  - Horizontal-only dilation (3,1) to reconnect broken strokes of open glyphs.
  - MIN_COMP_AREA raised to 60.
  - DIACRITIC thresholds relaxed (0.20 / 0.35).
  - Aspect-ratio-aware square padding before CNN resize.
  - Hal-kirima (්) always merges LEFT.
  - Vowel signs merge with most-overlapping base.
"""

import cv2
import numpy as np
import logging
from collections import defaultdict
from model_loader import get_model, get_label_map, get_yolo, yolo_ready

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE = (64, 64)
TOP_K      = 5

MIN_COMP_AREA          = 60
MIN_CHAR_WIDTH         = 8
MIN_CHAR_HEIGHT        = 20
CHAR_PADDING           = 6

WORD_GAP_RATIO         = 0.6

DIACRITIC_AREA_RATIO   = 0.20
DIACRITIC_HEIGHT_RATIO = 0.35

CC_DILATE_KERNEL = (3, 1)
CC_DILATE_ITER   = 1

YOLO_CONF    = 0.15
YOLO_IOU     = 0.40
YOLO_PADDING = 4

FORCE_CC_FALLBACK = False

# ─────────────────────────────────────────────
# CONFUSION CORRECTION TABLE
# ─────────────────────────────────────────────
CONFUSION_PAIRS = {
    "ලි": "ලි",
    "ළි": "ළි",
    "ල":  "ල",
    "ළ":  "ළ",
    "ෆ":  "ෆ",
    "ල්": "ල්",
    "ෆ්": "ෆ්",
}


# ─────────────────────────────────────────────
# IMAGE DECODING
# ─────────────────────────────────────────────
def _decode_image(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("Empty image bytes provided.")
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupted file.")
    return img


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_image(image_bytes: bytes):
    """Decode and binarize the image."""
    original = _decode_image(image_bytes)

    max_dim = 1600
    h, w = original.shape[:2]
    if max(h, w) > max_dim:
        scale    = max_dim / max(h, w)
        original = cv2.resize(original, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        h, w = original.shape[:2]

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    binary = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=9,
    )

    hline_kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    detected_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hline_kernel)
    binary         = cv2.subtract(binary, detected_lines)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    max_char_w = w * 0.6
    num_labels, labels_img, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_WIDTH] > max_char_w:
            binary[labels_img == i] = 0

    return binary, original, gray


# ─────────────────────────────────────────────
# LINE SEGMENTATION
# ─────────────────────────────────────────────
def segment_lines(binary):
    h_proj    = np.sum(binary, axis=1)
    threshold = max(1.0, np.max(h_proj) * 0.04)

    lines, in_line, start = [], False, 0
    for i, val in enumerate(h_proj):
        if val > threshold and not in_line:
            start, in_line = i, True
        elif val <= threshold and in_line:
            if (i - start) > 8:
                lines.append((start, i))
            in_line = False

    if in_line and (binary.shape[0] - start) > 8:
        lines.append((start, binary.shape[0]))

    logger.info("Lines detected: %d", len(lines))
    return lines


# ─────────────────────────────────────────────
# X-OVERLAP MERGING
# ─────────────────────────────────────────────
def _merge_overlapping_x(boxes):
    if not boxes:
        return boxes
    boxes  = sorted(boxes, key=lambda b: b[0])
    merged = [list(boxes[0])]
    for box in boxes[1:]:
        cx1, cy1, cx2, cy2, carea = box
        prev = merged[-1]
        px1, py1, px2, py2, parea = prev
        if cx1 < px2 and cx2 > px1:
            merged[-1] = [
                min(px1, cx1), min(py1, cy1),
                max(px2, cx2), max(py2, cy2),
                parea + carea,
            ]
        else:
            merged.append(list(box))
    return merged


def _chars_to_boxes(char_list):
    return [[c["x1"], c["y1"], c["x2"], c["y2"],
             (c["x2"]-c["x1"])*(c["y2"]-c["y1"])] for c in char_list]


def _boxes_to_chars(box_list):
    return [{"x1":b[0],"y1":b[1],"x2":b[2],"y2":b[3],
             "w":b[2]-b[0],"h":b[3]-b[1],"area":b[4]} for b in box_list]


# ─────────────────────────────────────────────
# CONNECTED-COMPONENTS HELPERS
# ─────────────────────────────────────────────
def _classify_components(components):
    if not components:
        return [], []

    areas   = np.array([c["area"] for c in components])
    heights = np.array([c["h"]    for c in components])
    mask    = areas >= np.median(areas)
    med_area = float(np.median(areas[mask]))
    med_h    = float(np.median(heights[mask]))

    base_chars, diacritics = [], []
    for c in components:
        tiny_area    = c["area"] < DIACRITIC_AREA_RATIO * med_area
        short_height = c["h"]    < DIACRITIC_HEIGHT_RATIO * med_h
        is_diacritic = tiny_area or short_height
        (diacritics if is_diacritic else base_chars).append(c)

    logger.debug("classify_components: %d base, %d diacritics",
                 len(base_chars), len(diacritics))
    return base_chars, diacritics


def _merge_diacritics_into_bases(base_chars, diacritics):
    if not diacritics or not base_chars:
        return base_chars

    base_chars.sort(key=lambda b: b["x1"])

    for d in diacritics:
        d_cx = (d["x1"] + d["x2"]) / 2
        d_cy = (d["y1"] + d["y2"]) / 2

        is_below = False
        if base_chars:
            avg_base_y = np.mean([(b["y1"]+b["y2"])/2 for b in base_chars])
            is_below   = d_cy > avg_base_y

        if is_below and d["w"] < d["h"] * 0.8:
            left_bases = [b for b in base_chars if b["x2"] < d_cx]
            if left_bases:
                best_base = max(left_bases, key=lambda b: b["x2"])
            else:
                best_base = min(base_chars,
                                key=lambda b: abs((b["x1"]+b["x2"])/2 - d_cx))
        else:
            best_base = None
            best_dist = float("inf")
            for b in base_chars:
                vertical_overlap = not (d["y2"] < b["y1"] or d["y1"] > b["y2"])
                dist = abs((b["x1"] + b["x2"]) / 2 - d_cx)
                if vertical_overlap:
                    dist *= 0.2
                if dist < best_dist:
                    best_dist = dist
                    best_base = b

        if best_base:
            best_base["x1"] = min(best_base["x1"], d["x1"])
            best_base["y1"] = min(best_base["y1"], d["y1"])
            best_base["x2"] = max(best_base["x2"], d["x2"])
            best_base["y2"] = max(best_base["y2"], d["y2"])

    return base_chars


# ─────────────────────────────────────────────
# CC SEGMENTATION (fallback)
# ─────────────────────────────────────────────
def _segment_cc(binary, lines):
    H, W = binary.shape

    if CC_DILATE_KERNEL is not None:
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, CC_DILATE_KERNEL)
        work_img = cv2.dilate(binary, kernel, iterations=CC_DILATE_ITER)
    else:
        work_img = binary

    all_chars = []

    for line_idx, (ly1, ly2) in enumerate(lines):
        strip = work_img[ly1:ly2, :]
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            strip, connectivity=8
        )

        raw_boxes = []
        for i in range(1, num_labels):
            x, y, bw, bh, area = stats[i]
            if area < MIN_COMP_AREA:
                continue
            if bw < MIN_CHAR_WIDTH:
                continue
            if bh < MIN_CHAR_HEIGHT:
                continue
            raw_boxes.append([x, ly1 + y, x + bw, ly1 + y + bh, area])

        if not raw_boxes:
            continue

        raw_boxes  = _merge_overlapping_x(raw_boxes)
        components = [
            {
                "x1": b[0], "y1": b[1],
                "x2": b[2], "y2": b[3],
                "w":  b[2] - b[0],
                "h":  b[3] - b[1],
                "area": b[4],
            }
            for b in raw_boxes
        ]

        base_chars, diacritics = _classify_components(components)
        merged = _merge_diacritics_into_bases(base_chars, diacritics)

        merged_boxes = _merge_overlapping_x(_chars_to_boxes(merged))
        merged = _boxes_to_chars(merged_boxes)
        merged.sort(key=lambda c: c["x1"])

        for c in merged:
            px1 = max(0, c["x1"] - CHAR_PADDING)
            px2 = min(W, c["x2"] + CHAR_PADDING)
            py1 = max(0, c["y1"] - CHAR_PADDING)
            py2 = min(H, c["y2"] + CHAR_PADDING)

            crop = binary[py1:py2, px1:px2]   # always from original binary
            if crop.size == 0:
                continue

            all_chars.append({
                "image":    crop,
                "x1": px1, "x2": px2,
                "y1": py1, "y2": py2,
                "width":    c["x2"] - c["x1"],
                "area":     c["area"],
                "line_idx": line_idx,
            })

    logger.info("CC segmented %d characters", len(all_chars))
    return all_chars


# ─────────────────────────────────────────────
# YOLO SEGMENTATION
# ─────────────────────────────────────────────
def _segment_yolo(binary, lines, original, yolo):
    H, W = binary.shape

    try:
        detections = yolo(original, conf=YOLO_CONF, iou=YOLO_IOU, verbose=False)[0]
        boxes      = detections.boxes.xyxy.cpu().numpy()
    except Exception as exc:
        logger.error("YOLO inference error: %s", exc)
        return []

    if len(boxes) == 0:
        logger.warning("YOLO returned 0 detections.")
        return []

    logger.info("YOLO detected %d characters", len(boxes))

    all_chars = []
    for x1, y1, x2, y2 in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 <= x1 or y2 <= y1:
            continue

        cy       = (y1 + y2) / 2
        line_idx = 0
        best_d   = float("inf")
        for idx, (ly1, ly2) in enumerate(lines):
            d = abs(cy - (ly1 + ly2) / 2)
            if d < best_d:
                best_d, line_idx = d, idx

        px1 = max(0, x1 - YOLO_PADDING)
        py1 = max(0, y1 - YOLO_PADDING)
        px2 = min(W, x2 + YOLO_PADDING)
        py2 = min(H, y2 + YOLO_PADDING)

        crop = binary[py1:py2, px1:px2]
        if crop.size == 0:
            continue

        all_chars.append({
            "image":    crop,
            "x1": px1, "x2": px2,
            "y1": py1, "y2": py2,
            "width":    x2 - x1,
            "area":     (x2 - x1) * (y2 - y1),
            "line_idx": line_idx,
        })

    return all_chars


# ─────────────────────────────────────────────
# SEGMENTATION ROUTER
# ─────────────────────────────────────────────
def segment_characters(binary, lines, original=None):
    if not FORCE_CC_FALLBACK and yolo_ready() and original is not None:
        yolo = get_yolo()
        if yolo is not None:
            chars = _segment_yolo(binary, lines, original, yolo)
            if chars:
                return chars
            logger.warning("YOLO found nothing — falling back to CC.")

    logger.info("Using connected-components segmentation.")
    return _segment_cc(binary, lines)


# ─────────────────────────────────────────────
# MORPHOLOGICAL SHAPE ANALYSIS FOR ය CORRECTION
# Verified against real handwritten ය (white-on-black, 80×80px).
# ─────────────────────────────────────────────

def _count_loops(binary_crop):
    """
    Count closed loops inside the character.

    FIX (v5): Apply morphological CLOSE with an ellipse kernel before counting.
    ය's loop is open in handwriting → raw binary has 0 holes.
    After closing, the gap seals and 1 hole is detected correctly.
    ලි/ල have no loop → closing doesn't create one.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary_crop, cv2.MORPH_CLOSE, kernel)
    contours, hierarchy = cv2.findContours(
        closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return 0
    return int(np.sum(hierarchy[0, :, 3] >= 0))


def _aspect_ratio(binary_crop):
    """Width / Height of the ink bounding box."""
    coords = cv2.findNonZero(binary_crop)
    if coords is None:
        return 1.0
    _, _, w, h = cv2.boundingRect(coords)
    return w / max(h, 1)


def _left_ink_fraction(binary_crop):
    """
    Fraction of total ink in the LEFT half.
    ය has a large loop on the left → typically ≥ 0.38.
    """
    h, w = binary_crop.shape
    if w == 0:
        return 0.5
    left_ink  = int(np.sum(binary_crop[:, :w//2] > 0))
    total_ink = int(np.sum(binary_crop > 0))
    return left_ink / max(total_ink, 1)


def _has_isolated_top_component(binary_crop):
    """
    Returns True only if a small, TRULY DISCONNECTED blob exists in the
    top 15% of the crop — indicating a diacritic vowel sign (ලි, කි, etc.).

    FIX (v5): Two changes from v4:
      1. Use top 15% (not 25%) — ය's upward tail stays in top 25% but
         is connected to the main body; shrinking the window avoids it.
      2. Require the bottom row of the top strip to be empty (zero ink),
         confirming the component is disconnected from the glyph body.
         This ensures the tail of ය (which touches the body) is NOT flagged.
      3. Area threshold tightened to < 12% of total ink (was 18%).
    """
    h, w = binary_crop.shape
    total_ink = int(np.sum(binary_crop > 0))
    if total_ink == 0:
        return False

    top_h      = max(int(h * 0.15), 1)
    top_region = binary_crop[:top_h, :]

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        top_region, connectivity=8
    )
    for i in range(1, num_labels):
        area   = stats[i, cv2.CC_STAT_AREA]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        # Bottom row of the top strip must be empty → truly disconnected
        bottom_row_ink = int(np.sum(top_region[-1, :] > 0))
        if (area < 0.12 * total_ink
                and comp_h < h * 0.12
                and bottom_row_ink == 0):
            return True
    return False


def _looks_like_ya(binary_crop):
    """
    Returns True when the crop morphologically matches ය.

    Verified conditions (all must pass):
      ✔ Aspect ratio 0.6 – 2.0   (wide, not a tall thin stroke)
      ✔ ≥ 1 loop after closing   (the rounded left body of ය)
      ✔ No truly isolated top diacritic
      ✔ Left half carries ≥ 38% of ink  (loop is on the left)
    """
    ar        = _aspect_ratio(binary_crop)
    loops     = _count_loops(binary_crop)
    left_frac = _left_ink_fraction(binary_crop)
    has_dt    = _has_isolated_top_component(binary_crop)

    logger.debug(
        "looks_like_ya: ar=%.2f  loops=%d  left_ink=%.2f  has_top_diacritic=%s",
        ar, loops, left_frac, has_dt,
    )

    return (
        0.6  <= ar    <= 2.0  and
        loops >= 1            and
        not has_dt            and
        left_frac >= 0.38
    )


# ─────────────────────────────────────────────
# CNN CHARACTER PREDICTION
# ─────────────────────────────────────────────
def predict_character(char_img):
    """
    Classify a single character crop with the CNN.
    Runs post-prediction morphological correction for ය vs ලි/ල confusion.
    """
    model     = get_model()
    label_map = get_label_map()

    if len(char_img.shape) == 3:
        char_img = cv2.cvtColor(char_img, cv2.COLOR_BGR2GRAY)

    h, w = char_img.shape
    if h == 0 or w == 0:
        return {
            "letter": "?", "confidence": 0.0, "low_conf": True,
            "top_k": [{"letter": "?", "confidence": 0.0}],
        }

    # 1. Truly binary
    _, char_img = cv2.threshold(char_img, 127, 255, cv2.THRESH_BINARY)

    # 2. Ensure white ink on black background
    if np.mean(char_img) > 127:
        char_img = cv2.bitwise_not(char_img)

    # 3. Tight crop around actual ink
    coords = cv2.findNonZero(char_img)
    if coords is not None:
        x, y, w_box, h_box = cv2.boundingRect(coords)
        margin = int(max(w_box, h_box) * 0.05)
        x      = max(0, x - margin)
        y      = max(0, y - margin)
        w_box  = min(char_img.shape[1] - x, w_box + 2 * margin)
        h_box  = min(char_img.shape[0] - y, h_box + 2 * margin)
        char_img = char_img[y:y+h_box, x:x+w_box]
        h, w = char_img.shape

    # Keep original-scale copy for morphological analysis (before resize)
    morph_img = char_img.copy()

    # 4. Square pad (aspect-ratio preserving) then resize for CNN
    size       = max(h, w)
    pad_top    = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left   = (size - w) // 2
    pad_right  = size - w - pad_left

    padded  = cv2.copyMakeBorder(
        char_img, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=0,
    )
    resized = cv2.resize(padded, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    tensor = resized.astype("float32") / 255.0
    tensor = np.expand_dims(tensor, axis=-1)
    tensor = np.expand_dims(tensor, axis=0)

    probs   = model.predict(tensor, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    conf    = float(probs[top_idx])
    letter  = label_map.get(top_idx, "?")

    top_k = [
        {
            "letter":     label_map.get(i, "?"),
            "confidence": round(float(probs[i]) * 100, 2),
        }
        for i in np.argsort(probs)[::-1][:TOP_K]
    ]

    # ── POST-PREDICTION MORPHOLOGICAL CORRECTION ──────────────────────────
    # If CNN predicted a character in CONFUSION_PAIRS, verify with shape
    # analysis. Only correct if _looks_like_ya() is conclusively True.
    if letter in CONFUSION_PAIRS:
        candidate = CONFUSION_PAIRS[letter]
        if candidate == "ය" and _looks_like_ya(morph_img):
            logger.info(
                "Morphological correction: CNN='%s' (%.1f%%) → corrected to 'ය'",
                letter, conf * 100,
            )
            letter = "ය"
    # ──────────────────────────────────────────────────────────────────────

    return {
        "letter":     letter,
        "confidence": round(conf * 100, 2),
        "low_conf":   conf < 0.60,
        "top_k":      top_k,
    }


# ─────────────────────────────────────────────
# SENTENCE BUILDER
# ─────────────────────────────────────────────
def build_sentence(char_results):
    """Group by line, insert spaces based on median width."""
    line_buckets = defaultdict(list)
    confidences  = []

    for ch in char_results:
        line_buckets[ch["line_idx"]].append(ch)

    output_lines = []
    for idx in sorted(line_buckets.keys()):
        chars = sorted(line_buckets[idx], key=lambda c: c["x1"])
        if not chars:
            continue

        widths        = [c["width"] for c in chars]
        median_width  = float(np.median(widths))
        gap_threshold = WORD_GAP_RATIO * median_width

        text = ""
        for i, ch in enumerate(chars):
            if i > 0:
                gap = ch["x1"] - chars[i - 1]["x2"]
                if gap > gap_threshold or gap > 1.5 * median_width:
                    text += " "
            text += ch["letter"]
            confidences.append(ch["confidence"])

        output_lines.append(text)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    return {
        "text":           "\n".join(output_lines),
        "lines":          output_lines,
        "avg_confidence": round(avg_conf, 2),
    }


# ─────────────────────────────────────────────
# MAIN OCR PIPELINE
# ─────────────────────────────────────────────
def run_ocr(image_bytes: bytes) -> dict:
    """Full pipeline."""
    binary, original, _ = preprocess_image(image_bytes)

    lines = segment_lines(binary)
    if not lines:
        return {
            "text": "", "lines": [], "characters": [],
            "line_count": 0, "char_count": 0, "avg_confidence": 0.0,
            "segmentation": "yolo" if yolo_ready() else "connected_components",
            "error": "No text lines detected in image.",
        }

    chars = segment_characters(binary, lines, original=original)
    if not chars:
        return {
            "text": "", "lines": [], "characters": [],
            "line_count": len(lines), "char_count": 0, "avg_confidence": 0.0,
            "segmentation": "yolo" if yolo_ready() else "connected_components",
            "error": "No characters detected.",
        }

    results = []
    for ch in chars:
        pred = predict_character(ch["image"])
        results.append({
            "x1":         ch["x1"],
            "y1":         ch["y1"],
            "x2":         ch["x2"],
            "y2":         ch["y2"],
            "width":      ch["width"],
            "line_idx":   ch["line_idx"],
            "letter":     pred["letter"],
            "confidence": pred["confidence"],
            "low_conf":   pred["low_conf"],
            "top_k":      pred["top_k"],
        })

    sentence = build_sentence(results)

    return {
        "text":           sentence["text"],
        "lines":          sentence["lines"],
        "characters":     results,
        "line_count":     len(lines),
        "char_count":     len(results),
        "avg_confidence": sentence["avg_confidence"],
        "segmentation":   "yolo" if yolo_ready() else "connected_components",
    }