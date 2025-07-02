# utils/dimension.py
def estimate_cm_dimensions(pixel_width, pixel_height, dpi=96):
    # Convert pixels to cm (1 inch = 2.54 cm)
    inches_w = pixel_width / dpi
    inches_h = pixel_height / dpi
    return {
        "width": round(inches_w * 2.54, 2),
        "height": round(inches_h * 2.54, 2)
    }
