import cv2
import numpy as np
from PIL import Image


def detect_exact_pupil_and_iris(image_input):
    """
    Precision Pupil & Iris Circle Detector:
    1. Converts image to grayscale and applies Gaussian blur to suppress noise.
    2. Applies Adaptive Pupil Thresholding to isolate dark pupil region.
    3. Runs Hough Circle Transform to pinpoint exact pupil center (cx, cy) and radius r.
    4. Draws bright BLUE detection circle around the exact pupil/iris!
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        np_img = np.array(pil_img)
    else:
        np_img = cv2.imread(image_input)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    h, w = np_img.shape[:2]
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    annotated = np_img.copy()

    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 1. Dark Pupil Thresholding & Contour Enclosing Circle
    # Find lowest 10% brightness threshold (pupil is darkest area)
    min_val = np.min(blurred)
    thresh_val = min(100, min_val + 35)
    _, thresh = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pupil_found = False
    best_cx, best_cy, best_r = w // 2, h // 2, min(h, w) // 4

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < (h * w * 0.25):
            (cx, cy), r = cv2.minEnclosingCircle(cnt)
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity > 0.4 and 0.1 * min(h, w) < r < 0.4 * min(h, w):
                    best_cx, best_cy, best_r = int(cx), int(cy), int(r)
                    pupil_found = True
                    break

    if not pupil_found:
        # Hough Circles fallback
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(h, w) // 3,
            param1=50, param2=20, minRadius=int(min(h, w) * 0.08), maxRadius=int(min(h, w) * 0.40)
        )
        if circles is not None:
            circles = np.uint16(np.round(circles))
            best_cx, best_cy, best_r = int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])
            pupil_found = True

    iris_radius = int(best_r * 2.2)

    # Draw bright BLUE detection circles directly around the detected eye/iris (RGB: 0, 140, 255)
    cv2.circle(annotated, (best_cx, best_cy), iris_radius, (0, 140, 255), 4)  # Outer blue iris circle
    cv2.circle(annotated, (best_cx, best_cy), best_r, (0, 220, 255), 2)     # Inner pupil circle
    cv2.circle(annotated, (best_cx, best_cy), 4, (255, 0, 0), -1)           # Red center pupil dot

    # Crop detected eye region
    x1 = max(0, best_cx - iris_radius)
    y1 = max(0, best_cy - iris_radius)
    x2 = min(w, best_cx + iris_radius)
    y2 = min(h, best_cy + iris_radius)

    cropped = np_img[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = np_img

    return Image.fromarray(cropped), Image.fromarray(annotated), pupil_found


if __name__ == "__main__":
    img_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
    crop, ann, found = detect_exact_pupil_and_iris(img_path)
    ann.save("scratch/test_pupil_detection.jpg")
    print(f"Pupil Detected: {found}. Saved annotated image to scratch/test_pupil_detection.jpg")
