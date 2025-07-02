# utils/detector.py
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # Or use a fine-tuned poster model if available

def detect_poster(image_path):
    results = model(image_path)[0]
    detections = []
    for box in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = box
        if conf >= 0.5:  # threshold
            detections.append({
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "confidence": round(conf, 2)
            })
    return detections
