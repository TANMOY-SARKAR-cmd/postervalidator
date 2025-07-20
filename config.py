# config.py

# --- Detector Settings ---
POSTER_FINDER = {
    "min_area_ratio": 0.005,
    "min_rectangularity": 0.55,
    "aspect_ratio_range": (0.2, 5.0),
}

YOLO = {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.4,
}

# --- Dimension Estimation ---
KNOWN_DIMENSIONS_CM = {
    "bus": 320,
    "truck": 350,
    "car": 150,
    "traffic light": 120,
    "person": 170,
}

RELIABILITY = {
    "bus": 0.9,
    "truck": 0.85,
    "traffic light": 0.7,
    "car": 0.6,
    "person": 0.4,
}

FALLBACK_DPI = 96

# --- OCR Settings ---
OCR = {
    "languages": ['en', 'hi'],
    "confidence_threshold": 0.5,
}

# --- Logo Matcher Settings ---
LOGO_MATCHER = {
    "good_match_threshold": 0.7,
    "min_matches": 10,
    "cache_size": 32,
    "orb_features": 1000,
}
