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
    expected_texts = data.get("texts", [])
    expected_logos = data.get("logos", [])  # Optional, placeholder

    if not image_url:
        return jsonify({"error": "imageUrl required"}), 400

    # Fetch the image
    response = requests.get(image_url, stream=True)
    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch image"}), 400

    image_path = "temp.jpg"
    with open(image_path, 'wb') as f:
        f.write(response.content)

    # Run poster detection
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

        # Optional: match OCR text if provided
        if expected_texts:
            ocr_results = detect_text(image_path)
            found_texts = [item["text"].lower() for item in ocr_results]
            matched = any(expected.lower() in found_texts for expected in expected_texts)
            result["matchedTexts"] = matched
            result["matchedTextList"] = found_texts  # for debugging

                # Optional: add logo matching
        if expected_logos:
            matched_logos = []
            poster_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            for logo_url in expected_logos:
                try:
                    logo_res = requests.get(logo_url, stream=True)
                    if logo_res.status_code != 200:
                        continue
                    logo_path = "logo_temp.jpg"
                    with open(logo_path, "wb") as f:
                        f.write(logo_res.content)

                    logo_img = cv2.imread(logo_path, cv2.IMREAD_GRAYSCALE)
                    if logo_img is None or poster_img is None:
                        continue

                    result_match = cv2.matchTemplate(poster_img, logo_img, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_match)

                    if max_val > 0.8:
                        matched_logos.append(logo_url)

                except Exception as e:
                    print(f"Error matching logo: {logo_url} — {e}")
                    continue

            result["matchedLogos"] = len(matched_logos) > 0
            result["matchedLogoList"] = matched_logos

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
