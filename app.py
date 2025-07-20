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
        contains_poster = poster_info is not None
        result["containsPoster"] = contains_poster

        if not contains_poster:
            return jsonify(result)

        # Poster-dependent processing
        try:
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
        except Exception as e:
            return jsonify({"error": f"Dimension estimation failed: {str(e)}"}), 500

        # OCR Matching
        if expected_texts:
            try:
                ocr_results = detect_text(image_path, poster_info)
                found_texts = [item["text"].lower() for item in ocr_results]
                matched = any(any(expected.lower() in t for t in found_texts) for expected in expected_texts)
                result["matchedTexts"] = matched
                result["matchedTextList"] = found_texts
            except Exception as e:
                 result["matchedTexts"] = {"error": f"OCR failed: {str(e)}"}

        # Logo Matching
        if expected_logos:
            try:
                matched, matched_list = match_logos(image_path, expected_logos)
                result["matchedLogos"] = matched
                result["matchedLogoList"] = matched_list
            except Exception as e:
                result["matchedLogos"] = {"error": f"Logo matching failed: {str(e)}"}

    except Exception as e:
        return jsonify({"error": f"Poster detection failed: {str(e)}"}), 500

    return jsonify(result)

if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
    print("Flask server started.")
