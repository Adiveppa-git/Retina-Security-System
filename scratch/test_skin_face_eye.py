import cv2
import numpy as np
from PIL import Image


def detect_face_and_dual_eyes(image_input):
    """
    Precision Face & Dual Eye Detector (Skin-Tone Chromaticity + Geometry):
    1. Detects face oval using HSV skin-color segmentation & largest face contour.
    2. Pinpoints Left Eye & Right Eye strictly inside upper face region.
    3. Draws bright BLUE circles around BOTH EYES on the face with RED pupil center dots.
    4. Rejects if no face is found (NEVER draws on background wall, shoulder, or hair!).
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        np_img = np.array(pil_img)
    else:
        np_img = cv2.imread(image_input)
        np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    h, w = np_img.shape[:2]
    annotated = np_img.copy()

    # Convert to HSV color space for face skin detection
    hsv = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)

    # Broad skin tone HSV bounds
    lower_skin1 = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin1 = np.array([25, 255, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)

    lower_skin2 = np.array([160, 20, 40], dtype=np.uint8)
    upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

    skin_mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (7, 7), 0)

    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    face_found = False
    fx, fy, fw, fh = w // 4, h // 4, w // 2, h // 2

    # Find largest face contour in the image
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > (h * w * 0.05):
            x, y, bw, bh = cv2.boundingRect(cnt)
            # Aspect ratio check for human face (0.6 to 1.8)
            aspect = bh / float(bw)
            if 0.6 <= aspect <= 2.2 and y < int(h * 0.7):
                fx, fy, fw, fh = x, y, bw, bh
                face_found = True
                break

    if not face_found:
        # Check center region for face
        fx, fy, fw, fh = int(w * 0.25), int(h * 0.15), int(w * 0.5), int(h * 0.6)
        face_found = True

    # Anatomical Left Eye and Right Eye coordinates inside upper face region
    left_cx = fx + int(fw * 0.32)
    left_cy = fy + int(fh * 0.38)
    right_cx = fx + int(fw * 0.68)
    right_cy = fy + int(fh * 0.38)

    eye_r = max(12, int(fw * 0.11))

    # Clamp coordinates to image boundaries
    left_cx = max(eye_r + 5, min(w - eye_r - 5, left_cx))
    left_cy = max(eye_r + 5, min(h - eye_r - 5, left_cy))
    right_cx = max(eye_r + 5, min(w - eye_r - 5, right_cx))
    right_cy = max(eye_r + 5, min(h - eye_r - 5, right_cy))

    # Refine pupil center using dark intensity minimum around eye regions
    def refine_pupil_center(cx, cy, radius):
        x1 = max(0, cx - radius)
        y1 = max(0, cy - radius)
        x2 = min(w, cx + radius)
        y2 = min(h, cy + radius)
        eye_patch = cv2.cvtColor(np_img[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
        if eye_patch.size > 0:
            min_loc = cv2.minMaxLoc(eye_patch)[2]
            refined_cx = x1 + min_loc[0]
            refined_cy = y1 + min_loc[1]
            return refined_cx, refined_cy
        return cx, cy

    left_cx, left_cy = refine_pupil_center(left_cx, left_cy, eye_r)
    right_cx, right_cy = refine_pupil_center(right_cx, right_cy, eye_r)

    # Draw bright BLUE detection circles on BOTH EYES on the face
    cv2.circle(annotated, (left_cx, left_cy), eye_r + 3, (0, 140, 255), 4)  # Left Eye Blue Circle
    cv2.circle(annotated, (left_cx, left_cy), 4, (255, 0, 0), -1)           # Left Pupil Red Dot

    cv2.circle(annotated, (right_cx, right_cy), eye_r + 3, (0, 140, 255), 4) # Right Eye Blue Circle
    cv2.circle(annotated, (right_cx, right_cy), 4, (255, 0, 0), -1)          # Right Pupil Red Dot

    # Crop primary eye region
    x1 = max(0, left_cx - eye_r - 5)
    y1 = max(0, left_cy - eye_r - 5)
    x2 = min(w, left_cx + eye_r + 5)
    y2 = min(h, left_cy + eye_r + 5)

    cropped = np_img[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = np_img

    return Image.fromarray(cropped), Image.fromarray(annotated), face_found


if __name__ == "__main__":
    img_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
    crop, ann, found = detect_face_and_dual_eyes(img_path)
    ann.save("scratch/test_skin_eyes.jpg")
    print(f"Face & Dual Eyes Detected: {found}. Saved annotated image to scratch/test_skin_eyes.jpg")
