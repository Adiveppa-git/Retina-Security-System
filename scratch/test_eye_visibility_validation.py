import cv2
import numpy as np
from PIL import Image


def validate_eye_visibility(crop_gray):
    """
    Validates if a real human eye/pupil is actually visible in the crop:
    1. Checks intensity contrast (standard deviation). Hair/phone case has low/flat contrast.
    2. Checks for dark pupil circle presence.
    Returns: True if real eye is present, False if covered by phone/hand/hair.
    """
    if crop_gray is None or crop_gray.size == 0:
        return False

    std_dev = np.std(crop_gray)
    min_val = np.min(crop_gray)
    mean_val = np.mean(crop_gray)

    # Real human eyes have high local contrast (white sclera + dark pupil/iris)
    # A phone case, solid hair patch, or wall has uniform texture (low contrast or extreme dark/light)
    if std_dev < 18.0:
        print(f"[Eye Visibility Validation] REJECTED: Low Contrast (std_dev={std_dev:.2f} < 18.0). Eyes Covered!")
        return False

    # Check for dark pupil region relative to surrounding sclera
    if min_val > (mean_val - 25.0):
        print(f"[Eye Visibility Validation] REJECTED: No Dark Pupil Circle (min_val={min_val}, mean_val={mean_val:.2f}). Eyes Covered!")
        return False

    return True


if __name__ == "__main__":
    # Test on a hair/phone crop (low contrast or uniform) vs real eye crop
    blank_phone_crop = np.full((100, 100), 50, dtype=np.uint8)  # Phone back
    hair_crop = np.random.randint(20, 40, (100, 100), dtype=np.uint8)

    print("Phone Crop Valid:", validate_eye_visibility(blank_phone_crop))
    print("Hair Crop Valid :", validate_eye_visibility(hair_crop))
