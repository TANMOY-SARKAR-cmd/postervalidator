# app.py
from flask import Flask, request, jsonify
import requests
import cv2
import os

from utils.detector import detect_poster
from utils.ocr import detect_text
from utils.dimension import estimate_cm_dimensions

app = Flask(__name__)

@app.route("/validate-image", methods=["POST"])
def validate_image():
    data = request.json
    image_url = data.get("imageUrl")

    if not image_url:
        return jsonify({"error": "imageUrl required"}), 400

    # Fetch the image
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch image"}), 400

    image_path = "temp.jpg"
    with open(image_path, 'wb') as f:
        f.write(response.content)

    # Detect posters
    detections = detect_poster(image_path)
    containsPoster = len(detections) > 0

    result = {
        "containsPoster": containsPoster
    }

    if containsPoster:
        box = detections[0]['bbox']
        dims = estimate_cm_dimensions(box[2], box[3])
        result.update({
            "boundingBox": {
                "x": box[0], "y": box[1],
                "width": box[2], "height": box[3]
            },
            "dimensionsCm": dims,
            "confidence": detections[0]["confidence"]
        })

    return jsonify(result)

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))