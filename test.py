import cv2
from utils.detector import detect_poster_and_objects
from utils.ocr import detect_text
from utils.dimension import estimate_poster_size
from utils.logo_matcher import match_logos

def test_pipeline():
    image_path = "temp.jpg"  # Make sure this image exists

    # Download a test image if it doesn't exist
    import requests
    import os
    if not os.path.exists(image_path):
        print("Downloading test image...")
        url = "https://www.westsystem.com/wp-content/uploads/2020/07/epoxy-works-35-summer-2012-repairing-a-severely-damaged-rudder-01.jpg"
        response = requests.get(url, stream=True, timeout=5)
        with open(image_path, 'wb') as f:
            f.write(response.content)

    print("1. Detecting poster and objects...")
    poster_info, objects = detect_poster_and_objects(image_path)
    if poster_info:
        print("  - Poster found:", poster_info)
    else:
        print("  - No poster found.")
        return

    print("2. Estimating poster size...")
    dimensions = estimate_poster_size(poster_info, objects)
    print("  - Dimensions:", dimensions)

    print("3. Detecting text...")
    ocr_results = detect_text(image_path, poster_info)
    print("  - OCR results:", ocr_results)

    print("4. Matching logos...")
    logos_to_match = []  # Add some logo URLs here if you want to test this
    matched, matched_list = match_logos(image_path, logos_to_match)
    print("  - Matched logos:", matched, matched_list)

if __name__ == "__main__":
    test_pipeline()
