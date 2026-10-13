import cv2
import numpy as np
from PIL import Image


def is_eye_covered_or_invalid(crop_rgb):
    """
    Validates if a real human eye/pupil exists in the cropped region.
    Rejects smooth phone backs, wall cables, furniture, and hands.
    """
    if crop_rgb is None or crop_rgb.size == 0:
        return True

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    std_dev = np.std(gray)
    min_val = np.min(gray)
    mean_val = np.mean(gray)

    # Real human eye region has contrast between white sclera and dark pupil/iris:
    # std_dev >= 15.0 and dark pupil (min_val < mean_val - 20.0)
    if std_dev < 14.0 or min_val > (mean_val - 18.0):
        return True  # Covered / Invalid

    return False
