# utils/detector.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO once
model = YOLO("yolov8n.pt")

# Average height in cm for scale references
KNOWN_DIMENSIONS_CM = {
    "bus": 320,
    "truck": 350,
    "car": 150,
    "traffic light": 120,
    "person": 170,
}

# Heuristic reliability weights (tune as needed)
RELIABILITY = {
    "bus": 0.9,
    "truck": 0.85,
    "traffic light": 0.7,
    "car": 0.6,
    "person": 0.4,
}


def _find_poster_region(image,
                        min_area_ratio: float = 0.005,
                        min_rectangularity: float = 0.55,
                        aspect_ratio_range=(0.2, 5.0)):
    """
    Color-agnostic structural poster finder.
    Returns (x, y, w, h) or None.
    """
    h_img, w_img = image.shape[:2]
    img_area = float(h_img * w_img)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge map
    edges = cv2.Canny(gray, 50, 150)

    # Close gaps so contours form better rectangles
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = None
    best_score = -1.0

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        box_area = float(w * h)
        if box_area < img_area * min_area_ratio:
            continue  # too small

        contour_area = float(cv2.contourArea(c))
        if box_area <= 0:
            continue

        rectangularity = contour_area / box_area  # 1.0 = perfect fill
        if rectangularity < min_rectangularity:
            continue

        ar = w / float(h) if h > 0 else 999.0
        if not (aspect_ratio_range[0] <= ar <= aspect_ratio_range[1]):
            continue

        # Score: prefer large & well-filled rectangles
        score = (box_area / img_area) * rectangularity
        if score > best_score:
            best = (x, y, w, h)
            best_score = score

    return best


def detect_poster_and_objects(image_path):
    """
    Detects a poster region (color-agnostic) + YOLO objects usable for scaling.
    Returns (poster_info or None, objects list).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    # --- Poster detection (no yellow mask) ---
    poster_box = _find_poster_region(image)
    if poster_box is None:
        return None, []  # no poster-like region found

    x, y, w, h = poster_box
    poster_info = {
        "x": int(x),
        "y": int(y),
        "w": int(w),
        "h": int(h),
        "center": (x + w / 2.0, y + h / 2.0),
    }

    # --- YOLO object detection for scaling references ---
    detections = model(image)[0]
    objects = []
    for box in detections.boxes:
        name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        if name in KNOWN_DIMENSIONS_CM and conf > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            w_obj = float(x2 - x1)
            h_obj = float(y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            objects.append({
                "name": name,
                "confidence": conf,
                "bbox": (float(x1), float(y1), w_obj, h_obj),
                "center": (cx, cy),
            })

    return poster_info, objects
