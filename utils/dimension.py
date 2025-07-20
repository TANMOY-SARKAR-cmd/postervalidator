# utils/dimension.py
import math
import config

def estimate_poster_size(poster, objects):
    best = None
    score = -1
    px, py = poster["center"]
    for obj in objects:
        ox, oy = obj["center"]
        dist = math.hypot(px - ox, py - oy)
        prox = 1 / (1 + dist / 100)
        s = config.RELIABILITY[obj["name"]] * obj["confidence"] * prox
        if s > score:
            best, score = obj, s

    if best:
        ref_h_px = best["bbox"][3]
        ref_h_cm = config.KNOWN_DIMENSIONS_CM[best["name"]]
        scale = ref_h_px / ref_h_cm
        width_cm = poster["w"] / scale
        height_cm = poster["h"] / scale
        return {
            "cm": {"width": round(width_cm, 2), "height": round(height_cm, 2)},
            "scale": round(scale, 2),
            "fallback": False
        }
    else:
        # fallback using DPI conversion
        fallback_scale = config.FALLBACK_DPI / 2.54
        width_cm = poster["w"] / fallback_scale
        height_cm = poster["h"] / fallback_scale
        return {
            "cm": {"width": round(width_cm, 2), "height": round(height_cm, 2)},
            "scale": round(fallback_scale, 2),
            "fallback": True
        }
