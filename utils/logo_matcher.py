# utils/logo_matcher.py
import cv2
import requests
import numpy as np
from functools import lru_cache
import config

# Initialize the ORB detector
orb = cv2.ORB_create(nfeatures=config.LOGO_MATCHER["orb_features"])

@lru_cache(maxsize=config.LOGO_MATCHER["cache_size"])
def _get_logo_features(logo_url):
    """
    Downloads a logo, computes, and returns its keypoints and descriptors.
    Caches the results.
    """
    try:
        res = requests.get(logo_url, stream=True, timeout=5)
        if res.status_code != 200:
            return None, None

        file_bytes = np.asarray(bytearray(res.content), dtype=np.uint8)
        logo_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if logo_img is None:
            return None, None

        # Add a border to the logo to improve feature detection
        logo_img = cv2.copyMakeBorder(logo_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        return orb.detectAndCompute(logo_img, None)
    except Exception:
        return None, None

def match_logos(image_path, logo_urls, good_match_threshold=config.LOGO_MATCHER["good_match_threshold"], min_matches=config.LOGO_MATCHER["min_matches"]):
    """
    Matches logos using ORB feature detection.
    """
    main_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if main_img is None:
        return False, []

    kp_main, des_main = orb.detectAndCompute(main_img, None)
    if des_main is None:
        return False, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matched_logos = []

    for logo_url in logo_urls:
        kp_logo, des_logo = _get_logo_features(logo_url)
        if des_logo is None:
            continue

        matches = bf.match(des_logo, des_main)

        # Apply ratio test as per Lowe's paper
        good_matches = [m for m in matches if m.distance < good_match_threshold * min(m.distance for m in matches)]

        if len(good_matches) > min_matches:
            matched_logos.append(logo_url)

    return len(matched_logos) > 0, matched_logos
