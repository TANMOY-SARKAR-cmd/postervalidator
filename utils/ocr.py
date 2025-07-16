import easyocr
import cv2

reader = easyocr.Reader(['en', 'hi'])  # You can add more languages as needed

def detect_text(image_path, poster_info):
    img = cv2.imread(image_path)
    x, y, w, h = poster_info["x"], poster_info["y"], poster_info["w"], poster_info["h"]
    cropped = img[y:y+h, x:x+w]
    results = reader.readtext(cropped)
    return [{"text": t, "conf": round(conf, 2)} for (bbox, t, conf) in results if conf > 0.5]
