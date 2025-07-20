import easyocr
import cv2
import config

reader = None

def _get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(config.OCR["languages"])
    return reader

def detect_text(image_path, poster_info, confidence_threshold=config.OCR["confidence_threshold"]):
    reader = _get_ocr_reader()
    img = cv2.imread(image_path)
    x, y, w, h = poster_info["x"], poster_info["y"], poster_info["w"], poster_info["h"]
    cropped = img[y:y+h, x:x+w]
    results = reader.readtext(cropped)
    return [{"text": t, "conf": round(conf, 2)} for (bbox, t, conf) in results if conf > confidence_threshold]
