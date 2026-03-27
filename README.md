Absolutely! Here’s a **fully formatted, developer-friendly Markdown README** for your Sinhala Handwritten OCR backend. It includes **tables, collapsible sections, and a quick-start guide** to make it professional and easy to navigate.

---

# Sinhala Handwritten OCR API

A Flask-based REST API for recognizing **Sinhala handwritten text** using a deep learning CNN model. The backend processes images, segments characters, predicts Sinhala letters, and reconstructs full sentences with confidence scores.

---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [How It Works](#how-it-works)
4. [API Endpoints](#api-endpoints)
5. [Quick Start](#quick-start)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)
8. [Future Improvements](#future-improvements)
9. [Tech Stack](#tech-stack)
10. [Author & Support](#author--support)

---

## Features

| Category                | Features                                                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **OCR Pipeline**        | CNN-based Sinhala character recognition, Noise removal & preprocessing, Automatic deskew, Character segmentation, Word & line reconstruction |
| **Smart Output**        | Confidence score per character, Top-K predictions, Low-confidence detection, Word gap detection, Multi-line text support                     |
| **Backend Engineering** | Model caching, Fast inference pipeline, REST API with JSON responses, Robust error handling, Health monitoring endpoint                      |

---

## Project Structure

```text
backend/
│
├── app.py              # Flask API (routes + server)
├── model_loader.py     # Load and cache ML model
├── ocr_engine.py       # Full OCR pipeline
│
├── models/
│   ├── sinhala_cnn.h5
│   └── label_map.json
│
├── requirements.txt
└── README.md
```

---

## How It Works

<details>
<summary>Click to expand OCR Pipeline Flow</summary>

1. **Image Input**
2. **Preprocessing**
3. **Line Segmentation**
4. **Character Segmentation**
5. **CNN Prediction**
6. **Sentence Reconstruction**
7. **JSON Response**

</details>

### 1. Preprocessing

* Convert image to grayscale
* Remove noise using Gaussian blur
* Adaptive thresholding for uneven lighting
* Morphological operations to clean strokes
* Deskewing to correct tilted handwriting

### 2. Line Segmentation

* Uses horizontal projection profile
* Automatically detects text lines
* Filters thin/noisy lines

### 3. Character Segmentation

* Uses Connected Component Analysis
* Filters noise based on width, height, and area
* Merges nearby components (important for Sinhala diacritics)

### 4. Character Prediction

* Resize characters to 64x64 pixels
* Normalize and pass through CNN
* Returns:

  * Predicted Sinhala letter
  * Confidence score
  * Top-K alternative predictions

### 5. Sentence Reconstruction

* Sort characters left → right
* Detect word gaps
* Combine into words, lines, and full sentences

---

## API Endpoints

| Endpoint          | Method | Description                   | Input                                 | Response                                      |
| ----------------- | ------ | ----------------------------- | ------------------------------------- | --------------------------------------------- |
| `/health`         | GET    | Check server and model status | None                                  | JSON status, model_ready, num_classes         |
| `/predict`        | POST   | Predict from image upload     | `file: <image>`                       | JSON with text, lines, characters, confidence |
| `/predict/base64` | POST   | Predict from base64 image     | `{"image_base64": "<base64 string>"}` | JSON with text, lines, characters, confidence |

**Example Response:**

```json
{
  "text": "සිංහල වාක්‍යය",
  "lines": ["සිංහල වාක්‍යය"],
  "line_count": 1,
  "char_count": 8,
  "avg_confidence": 87.3,
  "characters": [
    {
      "letter": "ස",
      "confidence": 94.2,
      "low_conf": false,
      "top_k": [
        {"letter": "ස", "confidence": 94.2},
        {"letter": "ශ", "confidence": 3.1}
      ],
      "x1": 10,
      "x2": 42,
      "y1": 5,
      "y2": 60,
      "line_idx": 0
    }
  ]
}
```

---

## Quick Start

<details>
<summary>Click to expand Quick Start Instructions</summary>

1. **Clone Repository**

```bash
git clone https://github.com/your-username/sinhala-ocr.git
cd backend
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Add Model Files**

Place the following in `backend/models/`:

* `sinhala_cnn.h5`
* `label_map.json`

4. **Run the Server**

```bash
python app.py
```

Server runs at:

```
http://localhost:5000
```

5. **Test Endpoints**

* Health Check: `GET /health`
* Image Prediction: `POST /predict` or `POST /predict/base64`

</details>

---

## Error Handling

| Status Code | Meaning                                     |
| ----------- | ------------------------------------------- |
| 400         | Bad request (missing file or invalid input) |
| 413         | File too large (>16MB)                      |
| 415         | Unsupported file type                       |
| 422         | Processing error (invalid image)            |
| 500         | Internal server error                       |
| 503         | Model not loaded                            |

---

## Performance Tips

* Use clear, high-resolution images
* Avoid shadows or blur
* Keep text horizontally aligned
* Use dark ink on a light background

---

## Future Improvements

* Language model for sentence correction
* Transformer-based OCR (CRNN / TrOCR)
* Batch processing endpoint
* Bounding box visualization
* Model optimization (ONNX / TensorRT)
* PDF document OCR support
* Confidence analytics

---

## Tech Stack

* Python (Flask)
* TensorFlow / Keras
* OpenCV
* NumPy
* SciPy

---

## Author & Support

**Vinura Deelaka**
AI Enthusiast | Software Developer

If you find this project useful:

* Star the repository
* Contribute improvements
* Share with others

---
