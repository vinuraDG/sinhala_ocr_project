

---

# **Sinhala Handwritten OCR** 🖋️

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18.2.0-blue?logo=react)](https://reactjs.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A **full-stack OCR system** to recognize **handwritten Sinhala text** from images (phone camera or webcam) using **CNN + Flask backend + React frontend**.

---

## **🚀 Features**

* Capture Sinhala handwriting using **phone or webcam**.
* Preprocess images → segment → predict each character → combine into words.
* Returns **text + confidence per character**.
* Mobile and desktop friendly.
* Easily extendable for new datasets or models.

---

## **📐 System Architecture**

```
[React Frontend]
📷 Camera Capture → Preview → Send image
📋 Result Display → Text + Per-character confidence
       │ HTTP POST /predict
       ▼
[Flask Backend]
1. Receive image
2. Preprocess → grayscale, denoise, binarize
3. Segment lines & characters
4. CNN predicts each character
5. Combine → full sentence
6. Return JSON result
       │
       ▼
[CNN Model]
Input: 64x64 grayscale character
Output: Sinhala letter + confidence %
```

---

## **📂 Project Structure**

```bash
sinhala_ocr_project/
├── notebooks/          # Model training & testing notebooks
│   ├── 01_explore_dataset.ipynb
│   ├── 02_train_model.ipynb
│   └── 03_test_pipeline.ipynb
│
├── backend/            # Flask API + OCR engine
│   ├── app.py
│   ├── ocr_engine.py
│   ├── model_loader.py
│   ├── requirements.txt
│   └── models/
│       ├── sinhala_cnn.h5
│       └── label_map.json
│
└── frontend/           # React app
    ├── public/
    └── src/
        ├── App.jsx
        ├── components/
        │   ├── CameraCapture.jsx
        │   ├── ResultDisplay.jsx
        │   └── CharacterGrid.jsx
        └── api/
            └── ocrApi.js
```

---

## **⚡ Setup & Usage**

### **Backend (Flask)**

```bash
cd backend
pip install -r requirements.txt
python app.py
```

* Health check: `http://localhost:5000/health`
* `/predict` endpoint: POST an image → returns recognized text + confidence.

**Python Dependencies**
`flask`, `flask-cors`, `tensorflow`, `opencv-python-headless`, `numpy`, `pillow`

---

### **Frontend (React.js)**

```bash
cd frontend
npm install
npm start
```

* Open `http://localhost:3000`
* Components:

  * `CameraCapture.jsx` → Capture handwriting.
  * `ResultDisplay.jsx` → Display recognized text + confidence.
  * `CharacterGrid.jsx` → Optional character-level view.

**Example API Call (`ocrApi.js`)**

```javascript
import axios from "axios";

export async function recognizeText(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await axios.post("http://localhost:5000/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30000
  });

  return response.data;
}
```

---

## **📝 Usage Flow**

1. Open React frontend → Click **Open Camera** → Capture handwritten Sinhala.
2. Image sent to Flask `/predict`.
3. Backend returns recognized text + per-character confidence.
4. Display results → copy, analyze, or export.

---

## **💡 Tips for Developers**

* Run **backend first**, then frontend.
* Ensure **good lighting** & **flat paper** for best accuracy.
* Adjust `gap_threshold` in `ocr_engine.py` for better word segmentation.
* CNN accuracy depends on **dataset quality + data augmentation**.
* Easily extendable: add more handwriting datasets or fine-tune model.

---

## **🛠️ Tech Stack**

* **Backend:** Python, Flask, TensorFlow/Keras, OpenCV
* **Frontend:** React.js, Axios
* **Data:** Handwritten Sinhala characters dataset
* **Deployment:** Desktop & Mobile browsers

---

## **📖 References**

* TensorFlow CNN guide: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
* Flask documentation: [https://flask.palletsprojects.com/](https://flask.palletsprojects.com/)
* React docs: [https://reactjs.org/docs/getting-started.html](https://reactjs.org/docs/getting-started.html)

---

✅ **Developer-friendly Summary:**

* **Frontend:** Camera → API → display results.
* **Backend:** Receives image → preprocess → segment → predict characters → combine text.
* **CNN Model:** Recognizes Sinhala characters.
* **JSON Output:** Text + confidence → frontend visualizes.

---


