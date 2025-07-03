````markdown
# 🧠 Poster Validator API (Hackathon - Aerodek)

An intelligent image validation API built using Python and YOLOv8 for detecting posters or wall content in real-world scenes. This project is built for the **Aerodek Hackathon 2025** to support campaign validation, text/logo recognition, and real-world dimension estimation — without using any paid APIs or Docker.

---

## 🚀 Features

- ✅ Detects presence of posters in an image
- 🔍 Identifies bounding box and real-world dimensions in centimeters
- 🧾 Extracts and matches expected text via OCR
- 🖼️ Matches logo presence using OpenCV template matching
- ⚡ Lightweight and runs on CPU (YOLOv8n + EasyOCR)

---

## 🧪 Sample API Usage

### 📥 POST `/validate-image`

**Request Body (JSON)**:
```json
{
  "imageUrl": "https://arodek.com/image.jpg",
  "texts": ["arodek", "Campaign 2025"],
  "logos": ["https://example.com/logo.png"]
}
````

* `imageUrl`: URL of the image (required)
* `texts`: List of expected text phrases (optional)
* `logos`: List of logo image URLs (optional)

---

**Response**:

```json
{
  "containsPoster": true,
  "boundingBox": { "x": 45, "y": 60, "width": 300, "height": 500 },
  "dimensionsCm": { "width": 60.0, "height": 100.0 },
  "confidence": 0.94,
  "matchedTexts": true,
  "matchedTextList": ["Campaign 2025", "Arodek"],
  "matchedLogos": true,
  "matchedLogoList": ["https://example.com/logo.png"]
}
```

---

## 🧰 Tech Stack

* Python 3.12
* [YOLOv8n (Ultralytics)](https://github.com/ultralytics/ultralytics)
* OpenCV
* EasyOCR
* Flask

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/poster-validator-api.git
cd poster-validator-api

# 2. Create virtual env (optional)
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the server
python app.py
```

The server will run on: `http://localhost:3000`

---

## 📁 Project Structure

```
├── app.py                 # Main Flask app
├── yolov8n.pt             # YOLOv8n weights
├── devserver.sh           # Dev run script (optional)
├── utils/                 # Detection, OCR, dimensions
│   ├── detector.py
│   ├── dimension.py
│   └── ocr.py
├── requirements.txt
├── README.md
└── temp.jpg               # Temp downloaded image
```

---

## ⚠️ Limitations

* Using base YOLOv8n model (not trained on posters)
* Poster detection depends on heuristics or needs a custom-trained model
* Logo matching is basic (template match, not scale/rotation robust)

---

## 📌 License & Credits

* This project is built for educational and non-commercial purposes under the **Aerodek Hackathon 2025**.
* YOLOv8 by [Ultralytics](https://github.com/ultralytics/ultralytics)
* OCR powered by [EasyOCR](https://github.com/JaidedAI/EasyOCR)

---

## 👨‍💻 Author

Team Name- **Daredevil Dynamo**

**Tanmoy Sarkar**  **Sourin Rom**

Student & Developer
🚀 Hackathon enthusiast | 💡 Passionate about AI & Computer Vision

```
```
