# app.py
from flask import Flask, request, jsonify
import requests
import os
import cv2

from utils.detector import detect_poster_and_objects
from utils.ocr import detect_text
from utils.dimension import estimate_poster_size
from utils.logo_matcher import match_logos

app = Flask(__name__)

@app.route("/validate-image", methods=["POST"])
def validate_image():
    data = request.json
    image_url = data.get("imageUrl")
    expected_texts = data.get("texts", [])
    expected_logos = data.get("logos", [])

    if not image_url:
        return jsonify({"error": "imageUrl required"}), 400

    try:
        response = requests.get(image_url, stream=True, timeout=5)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image"}), 400

        image_path = "temp.jpg"
        with open(image_path, 'wb') as f:
            f.write(response.content)
    except Exception as e:
        return jsonify({"error": f"Exception while downloading image: {str(e)}"}), 500

    result = {}

    try:
        poster_info, objects = detect_poster_and_objects(image_path)
        containsPoster = poster_info is not None
        result["containsPoster"] = containsPoster

        if containsPoster:
            dimensions = estimate_poster_size(poster_info, objects)
            result.update({
                "boundingBox": {
                    "x": poster_info["x"], "y": poster_info["y"],
                    "width": poster_info["w"], "height": poster_info["h"]
                },
                "dimensionsCm": dimensions["cm"],
                "pixelsPerCm": dimensions["scale"],
                "fallbackUsed": dimensions["fallback"]
            })

            # OCR Matching
            if expected_texts:
                ocr_results = detect_text(image_path, poster_info)
                found_texts = [item["text"].lower() for item in ocr_results]
                matched = any(any(expected.lower() in t for t in found_texts) for expected in expected_texts)
                result["matchedTexts"] = matched
                result["matchedTextList"] = found_texts

            # Logo Matching
            if expected_logos:
                matched, matched_list = match_logos(image_path, expected_logos)
                result["matchedLogos"] = matched
                result["matchedLogoList"] = matched_list

    except Exception as e:
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
