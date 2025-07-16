#utils/logo_matcher.py
import cv2
import requests
import numpy as np

def match_logos(image_path, logo_urls, threshold=0.8):
    matched = []
    main_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if main_img is None:
        return False, []

    for logo_url in logo_urls:
        try:
            res = requests.get(logo_url, stream=True, timeout=5)
            if res.status_code != 200:
                continue

            file_bytes = np.asarray(bytearray(res.content), dtype=np.uint8)
            logo_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

            if logo_img is None:
                continue

            result = cv2.matchTemplate(main_img, logo_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            if max_val > threshold:
                matched.append(logo_url)
        except Exception:
            continue

    return len(matched) > 0, matched
