import cv2
import numpy as np
import logging
from collections import defaultdict
from model_loader import get_model, get_label_map

logger = logging.getLogger(__name__)

IMAGE_SIZE     = (64, 64)
MIN_CHAR_WIDTH  = 8
MIN_CHAR_HEIGHT = 10
MIN_CHAR_AREA   = 80
CHAR_PADDING    = 6
WORD_GAP_RATIO  = 1.8
TOP_K           = 3


# ── Preprocessing ──────────────────────────────────────────────

def _decode_image(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Cannot decode image. Send a valid JPEG or PNG.")
    return img


def preprocess_image(image_bytes):
    original = _decode_image(image_bytes)
    grey     = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(grey, (3, 3), 0)

    adaptive = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10)

    _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    adaptive_density = np.sum(adaptive > 0) / adaptive.size
    binary = adaptive if adaptive_density < 0.40 else otsu

    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Deskew
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) >= 50:
        rect  = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) >= 0.5:
            h, w    = binary.shape
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            binary  = cv2.warpAffine(binary, rot_mat, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
    return binary, original, grey


# ── Line segmentation ──────────────────────────────────────────

def segment_lines(binary, v_padding=6):
    h_proj    = np.sum(binary, axis=1).astype(np.float32)
    threshold = max(1.0, float(np.max(h_proj)) * 0.04)

    lines = []
    in_line = False
    start   = 0
    H       = binary.shape[0]

    for i, val in enumerate(h_proj):
        if val > threshold and not in_line:
            start   = i
            in_line = True
        elif val <= threshold and in_line:
            y1 = max(0, start - v_padding)
            y2 = min(H, i   + v_padding)
            if (y2 - y1) >= MIN_CHAR_HEIGHT:
                lines.append((y1, y2))
            in_line = False

    if in_line:
        lines.append((max(0, start - v_padding), H))

    return lines


# ── Character segmentation ─────────────────────────────────────

def segment_characters(binary, lines):
    all_chars = []
    W = binary.shape[1]

    for line_idx, (y1, y2) in enumerate(lines):
        line_strip = binary[y1:y2, :]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            line_strip, connectivity=8)

        raw_chars = []
        for label in range(1, num_labels):
            x    = stats[label, cv2.CC_STAT_LEFT]
            y    = stats[label, cv2.CC_STAT_TOP]
            w    = stats[label, cv2.CC_STAT_WIDTH]
            h    = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]

            if w < MIN_CHAR_WIDTH or h < MIN_CHAR_HEIGHT or area < MIN_CHAR_AREA:
                continue

            x1  = max(0, x     - CHAR_PADDING)
            x2  = min(W, x + w + CHAR_PADDING)
            cy1 = max(0, y1 + y     - CHAR_PADDING)
            cy2 = min(binary.shape[0], y1 + y + h + CHAR_PADDING)

            raw_chars.append({
                "image":   binary[cy1:cy2, x1:x2],
                "x1": x1, "x2": x2,
                "y1": cy1, "y2": cy2,
                "area": area, "width": w,
            })

        raw_chars.sort(key=lambda c: c["x1"])

        # Merge diacritics
        merged = [raw_chars[0]] if raw_chars else []
        for ch in raw_chars[1:]:
            prev = merged[-1]
            if ch["x1"] - prev["x2"] <= 4:
                new_x1 = min(prev["x1"], ch["x1"])
                new_x2 = max(prev["x2"], ch["x2"])
                new_y1 = min(prev["y1"], ch["y1"])
                new_y2 = max(prev["y2"], ch["y2"])
                merged[-1] = {
                    "x1": new_x1, "x2": new_x2,
                    "y1": new_y1, "y2": new_y2,
                    "area": prev["area"] + ch["area"],
                    "width": new_x2 - new_x1,
                    "image": binary[new_y1:new_y2, new_x1:new_x2],
                }
            else:
                merged.append(ch)

        for ch in merged:
            if ch["image"].size == 0:
                continue
            ch["line_idx"] = line_idx
            all_chars.append(ch)

    return all_chars


# ── Prediction ─────────────────────────────────────────────────

def predict_character(char_img):
    model     = get_model()
    label_map = get_label_map()

    h, w = char_img.shape[:2]
    size    = max(h, w)
    pad_top = (size - h) // 2
    pad_bot = size - h - pad_top
    pad_lft = (size - w) // 2
    pad_rgt = size - w - pad_lft
    padded  = cv2.copyMakeBorder(char_img, pad_top, pad_bot, pad_lft, pad_rgt,
                                  cv2.BORDER_CONSTANT, value=0)
    resized  = cv2.resize(padded, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    tensor   = resized.astype("float32") / 255.0
    tensor   = tensor.reshape(1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

    probs       = model.predict(tensor, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:TOP_K]

    best_idx    = int(top_indices[0])
    best_conf   = round(float(probs[best_idx]) * 100, 2)

    top_k_results = [
        {"letter": label_map.get(int(i), "?"), "confidence": round(float(probs[i]) * 100, 2)}
        for i in top_indices
    ]

    return {
        "letter":     label_map.get(best_idx, "?"),
        "confidence": best_conf,
        "top_k":      top_k_results,
        "low_conf":   best_conf < 60.0,
    }


# ── Sentence assembly ──────────────────────────────────────────

def build_sentence(char_results):
    lines_dict = defaultdict(list)
    for ch in char_results:
        lines_dict[ch["line_idx"]].append(ch)

    sentence_lines  = []
    all_confidences = []

    for line_idx in sorted(lines_dict.keys()):
        line_chars = sorted(lines_dict[line_idx], key=lambda c: c["x1"])
        avg_w = np.mean([c["width"] for c in line_chars]) if line_chars else 20

        line_text = ""
        for i, ch in enumerate(line_chars):
            if i > 0:
                gap = ch["x1"] - line_chars[i - 1]["x2"]
                if gap > avg_w * WORD_GAP_RATIO:
                    line_text += " "
            line_text += ch["letter"]
            all_confidences.append(ch["confidence"])

        sentence_lines.append(line_text)

    return {
        "text":           "\n".join(sentence_lines),
        "lines":          sentence_lines,
        "avg_confidence": round(float(np.mean(all_confidences)), 2) if all_confidences else 0.0,
    }


# ── Main entry ─────────────────────────────────────────────────

def run_ocr(image_bytes):
    binary, original, grey = preprocess_image(image_bytes)
    h, w = binary.shape

    lines = segment_lines(binary)
    if not lines:
        return {"text": "", "lines": [], "characters": [],
                "line_count": 0, "char_count": 0, "avg_confidence": 0.0,
                "error": "No text lines detected. Try better lighting."}

    all_chars = segment_characters(binary, lines)
    if not all_chars:
        return {"text": "", "lines": [], "characters": [],
                "line_count": len(lines), "char_count": 0, "avg_confidence": 0.0,
                "error": "No characters detected. Check image quality."}

    char_results = []
    for ch in all_chars:
        pred = predict_character(ch["image"])
        char_results.append({**ch, **pred})

    sentence = build_sentence(char_results)

    serializable_chars = [
        {
            "letter":     ch["letter"],
            "confidence": ch["confidence"],
            "low_conf":   ch["low_conf"],
            "top_k":      ch["top_k"],
            "x1": ch["x1"], "x2": ch["x2"],
            "y1": ch["y1"], "y2": ch["y2"],
            "line_idx": ch["line_idx"],
        }
        for ch in char_results
    ]

    return {
        "text":           sentence["text"],
        "lines":          sentence["lines"],
        "characters":     serializable_chars,
        "line_count":     len(lines),
        "char_count":     len(char_results),
        "avg_confidence": sentence["avg_confidence"],
        "image_size":     {"width": w, "height": h},
    }