# 🖋️ Sinhala Handwritten OCR System

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2.0-61DAFB?logo=react)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo)](https://ultralytics.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **full-stack OCR system** that recognizes **handwritten Sinhala text** from images captured via phone camera or webcam. Built with a **dual-engine architecture** — YOLOv8 for character detection and a custom CNN for classification — wrapped in a Flask API and React frontend.

---

## 📌 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [OCR Pipeline](#-ocr-pipeline)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Model Details](#-model-details)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Tips for Developers](#-tips-for-developers)
- [References](#-references)

---

## 🚀 Features

- 📷 Capture Sinhala handwriting via **phone camera or webcam**
- 🔍 **Dual segmentation engine** — YOLOv8 (primary) with Connected Components fallback
- 🧠 **CNN classifier** recognizes individual Sinhala characters at 64×64px
- 🔧 **Morphological post-correction** fixes common CNN confusions (ය, අ, ඇ, etc.)
- 📊 Returns **full text + per-character confidence scores + top-K predictions**
- 📱 Mobile and desktop friendly React frontend
- ⚡ Fast inference with automatic YOLO → CC fallback if no detections
- 🔣 Handles **diacritics, vowel signs, hal-kirima (්)** and right-side vowel signs (ා, ේ, ෝ)

---

## 📐 System Architecture

```
┌─────────────────────────────────────────────┐
│              React Frontend                  │
│  📷 Camera Capture → Preview → Send Image   │
│  📋 Result: Text + Per-character Confidence │
└────────────────────┬────────────────────────┘
                     │ HTTP POST /predict
                     ▼
┌─────────────────────────────────────────────┐
│              Flask Backend                   │
│                                             │
│  1. Receive image bytes                     │
│  2. Preprocess (grayscale, blur, binarize)  │
│  3. Remove horizontal lines                 │
│  4. Segment text lines (H-projection)       │
│  5. Detect characters (YOLO or CC)          │
│  6. Classify each character (CNN)           │
│  7. Morphological post-correction           │
│  8. Build sentence (spaces + line breaks)   │
│  9. Return JSON                             │
└────────────────────┬────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌──────────────────────┐
│   YOLOv8 Model   │  │  Connected Components │
│ Character detect │  │  Fallback Segmenter   │
│ (primary engine) │  │  (when YOLO fails)    │
└──────────────────┘  └──────────────────────┘
          │
          ▼
┌──────────────────┐
│    CNN Model     │
│  Input: 64×64px  │
│  Output: Sinhala │
│  letter + conf%  │
└──────────────────┘
```

---

## 🔬 OCR Pipeline

### Stage 1 — Preprocessing
- Resize large images to max 1600px
- Gaussian blur `(5×5)` to reduce noise
- Adaptive thresholding `(blockSize=31, C=9)` for binarization
- Morphological close `(2×2)` to reconnect broken strokes
- Remove horizontal ruled lines via morphological open

### Stage 2 — Line Segmentation
- Horizontal ink projection profile
- Threshold at 4% of max projection value
- Extracts individual text line strips

### Stage 3 — Character Segmentation (Dual Engine)
**YOLOv8 (Primary)**
- Runs on original color image
- `conf=0.15`, `iou=0.40`
- Falls back to CC if 0 detections

**Connected Components (Fallback)**
- Horizontal dilation `(3×1)` to reconnect broken strokes
- Filters by `MIN_AREA=60`, `MIN_W=8px`, `MIN_H=20px`
- Classifies diacritics vs base characters by area/height ratio
- Merges diacritics into base characters (top, bottom, right-side vowel signs)
- Right-side vowel signs (ා, ේ, ෝ, ො, ෞ) attach to the immediately preceding base

### Stage 4 — CNN Classification
- Tight ink crop → square pad → resize to `64×64`
- White ink on black background normalization
- CNN softmax → top predicted label + confidence
- **Top-K=5** predictions returned per character

### Stage 5 — Morphological Post-Correction
Fixes systematic CNN confusion errors using shape analysis:

| CNN Predicted | Corrected To | Validator |
|---|---|---|
| ලි, ළි, ල, ළ, ෆ, ල්, ෆ් | ය | `_looks_like_ya()` |
| ෂ, ශ | අ | `_looks_like_a()` |
| ඹ, බ | ඇ | `_looks_like_ae()` |

Each validator checks: loop count (after morphological close), aspect ratio, ink distribution, isolated top components.

### Stage 6 — Sentence Building
- Groups characters by line index
- Inserts spaces where gap > `0.6 × median_char_width`
- hal-kirima (්) always attaches left, no space inserted
- Right-side vowel signs attach left, no space inserted

---

## 🛠️ Tech Stack

### Backend
| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10 | Core language |
| Flask | 2.x | REST API server |
| Flask-CORS | latest | Cross-origin requests |
| TensorFlow / Keras | 2.x | CNN model training & inference |
| YOLOv8 (Ultralytics) | latest | Character detection (primary segmenter) |
| OpenCV | 4.x | Image preprocessing & morphology |
| NumPy | latest | Array operations |
| Pillow | latest | Image format handling |

### Frontend
| Technology | Version | Purpose |
|---|---|---|
| React.js | 18.2.0 | UI framework |
| Axios | latest | HTTP client for API calls |
| HTML5 Camera API | — | Webcam / phone camera access |

### Models
| Model | Format | Purpose |
|---|---|---|
| `sinhala_cnn.h5` | Keras HDF5 | Sinhala character classifier |
| YOLOv8 weights | `.pt` | Character bounding box detection |
| `label_map.json` | JSON | Index → Sinhala Unicode character mapping |

---

## 📂 Project Structure

```
Sinhala_OCR-Project/
│
├── backend/                        # Flask API + OCR engine
│   ├── app.py                      # Flask routes (/predict, /health)
│   ├── ocr_engine.py               # Full OCR pipeline
│   ├── model_loader.py             # CNN + YOLO model loader
│   ├── requirements.txt            # Python dependencies
│   └── models/
│       ├── sinhala_cnn.h5          # Trained CNN model
│       └── label_map.json          # Character index map
│       |-- sinhala_yolo.pt         # yolo model
├── frontend/                       # React frontend
│   ├── public/
│   └── src/
│       ├── App.jsx                 # Root component
│       ├── components/
│       │   ├── CameraCapture.jsx   # Camera capture component
│       │   ├── ResultDisplay.jsx   # OCR result display
│       │   └── CharacterGrid.jsx   # Per-character confidence grid
│       └── api/
│           └── ocrApi.js           # Axios API calls
│
├── notebooks/                     
│   ├── yolo-model-handwritten-letter-detection.ipynb # CNN model train
│   ├── sinhala-character-cnn-model.ipynb # YOLO model train
│   
│
├── .gitignore
└── README.md
```

---

## ⚡ Setup & Usage

### Prerequisites
- Python 3.10+
- Node.js 16+
- Git

---

### Backend (Flask)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

- Health check: `http://localhost:5000/health`
- Predict endpoint: `POST http://localhost:5000/predict`

**Python Dependencies**
```
flask
flask-cors
tensorflow
ultralytics        # YOLOv8
opencv-python-headless
numpy
pillow
```

---

### Frontend (React)

```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000`

> ⚠️ Start the **backend first**, then the frontend.

---

## 🤖 Model Details

### CNN Model (`sinhala_cnn.h5`)
- **Input:** 64×64 grayscale image (white ink on black background)
- **Architecture:** Custom CNN trained on handwritten Sinhala character dataset
- **Output:** Softmax probabilities over all Sinhala character classes
- **Confidence threshold:** `< 60%` flagged as `low_conf: true`
- **Top-K:** Returns top 5 predictions per character

### YOLOv8 Model
- **Purpose:** Detects and localizes individual character bounding boxes
- **Confidence:** `0.15` (low threshold to catch faint handwriting)
- **IoU:** `0.40`
- **Padding:** 4px added around each detected box before CNN crop
- **Fallback:** If YOLO returns 0 detections, Connected Components segmenter activates automatically

### Morphological Validators
Shape-based validators run **after** CNN prediction to correct systematic errors:

**`_looks_like_ya()`** — verifies ය:
- Aspect ratio `0.6–2.0`
- ≥1 closed loop (after ellipse morphological close)
- Left half ink ≥ 38%
- No isolated top diacritic

**`_looks_like_a()`** — verifies අ:
- Aspect ratio `0.5–1.8`
- ≥1 loop, balanced left/right ink `(0.25–0.75)`
- Bottom tail ink ≥ 10%
- No isolated top diacritic

**`_looks_like_ae()`** — verifies ඇ:
- Aspect ratio `0.5–2.0`
- ≥1 loop, balanced ink
- Top region ink ≥ 5% (the small mark above ඇ)

---

## 📡 API Reference

### `POST /predict`

**Request:**
```
Content-Type: multipart/form-data
Body: file = <image file>
```

**Response:**
```json
{
  "text": "අම්මා",
  "lines": ["අම්මා"],
  "characters": [
    {
      "x1": 10, "y1": 20, "x2": 60, "y2": 80,
      "letter": "අ",
      "confidence": 87.5,
      "low_conf": false,
      "top_k": [
        { "letter": "අ", "confidence": 87.5 },
        { "letter": "ආ", "confidence": 6.2 }
      ]
    }
  ],
  "line_count": 1,
  "char_count": 4,
  "avg_confidence": 82.3,
  "segmentation": "yolo"
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## ⚙️ Configuration

Key constants in `ocr_engine.py` you can tune:

```python
# Character segmentation
MIN_COMP_AREA   = 60     # Minimum pixel area to consider as a component
MIN_CHAR_WIDTH  = 8      # Minimum character width in pixels
MIN_CHAR_HEIGHT = 20     # Minimum character height in pixels
CHAR_PADDING    = 6      # Padding added around each character crop

# Word spacing
WORD_GAP_RATIO  = 0.6    # Gap > 0.6 × median_width → insert space

# Diacritic detection
DIACRITIC_AREA_RATIO   = 0.20   # Component < 20% median area → diacritic
DIACRITIC_HEIGHT_RATIO = 0.35   # Component < 35% median height → diacritic

# YOLO
YOLO_CONF = 0.15         # Lower = detect more (catches faint ink)
YOLO_IOU  = 0.40         # Overlap threshold for duplicate suppression

# Force CC segmentation even if YOLO is available
FORCE_CC_FALLBACK = False
```

---

## 💡 Tips for Developers

- **Lighting:** Good uniform lighting gives the best binarization results
- **Paper:** Flat, white, unruled paper works best (ruled lines are removed automatically)
- **Stroke width:** Medium pen strokes work better than very thin or very thick
- **Word spacing:** If words are being merged, lower `WORD_GAP_RATIO` (try `0.4`)
- **YOLO accuracy:** Retrain YOLO on your specific handwriting style for best results
- **CNN accuracy:** More diverse training data = better generalization
- **Debug crops:** Check `backend/debug_crops/` for individual character crops if results are wrong
- **Force CC mode:** Set `FORCE_CC_FALLBACK = True` to always use Connected Components (useful for debugging)

---

## 📖 References

- YOLOv8 / Ultralytics: https://docs.ultralytics.com
- TensorFlow CNN guide: https://www.tensorflow.org/tutorials
- Flask documentation: https://flask.palletsprojects.com
- OpenCV morphology: https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
- React docs: https://reactjs.org/docs/getting-started.html
- Sinhala Unicode block: https://unicode.org/charts/PDF/U0D80.pdf

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <b>Built with ❤️ for Sinhala language preservation and accessibility</b>
</div>
