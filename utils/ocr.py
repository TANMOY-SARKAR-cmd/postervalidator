# utils/ocr.py
import easyocr

reader = easyocr.Reader(['en'])

def detect_text(image_path):
    results = reader.readtext(image_path)
    texts = []
    for (bbox, text, conf) in results:
        if conf > 0.5:
            texts.append({
                "text": text,
                "confidence": round(conf, 2),
                "bbox": bbox
            })
    return texts
