# utils/detector.py
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Average height in cm
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

def detect_poster_and_objects(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, []

    poster = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(poster)
    poster_info = {"x": x, "y": y, "w": w, "h": h, "center": (x + w / 2, y + h / 2)}

    detections = model(image)[0]
    objects = []
    for box in detections.boxes:
        name = model.names[int(box.cls[0])]
        if name in KNOWN_DIMENSIONS_CM and float(box.conf[0]) > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            objects.append({
                "name": name, "confidence": float(box.conf[0]),
                "bbox": (x1, y1, w, h), "center": (cx, cy)
            })
    return poster_info, objects
