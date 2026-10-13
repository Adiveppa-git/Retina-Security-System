import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


def test_eye_detection(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(80, 80))
    print(f"Faces Detected: {len(faces)}")

    annotated = img.copy()

    if len(faces) > 0:
        fx, fy, fw, fh = faces[0]
        eye_roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(eye_roi_gray, scaleFactor=1.1, minNeighbors=3)
        print(f"Eyes Detected on Face: {len(eyes)}")

        for (ex, ey, ew, eh) in eyes:
            cx = fx + ex + ew // 2
            cy = fy + ey + eh // 2
            r = max(ew, eh) // 2
            # Draw blue circle around detected eye
            cv2.circle(annotated, (cx, cy), r + 4, (255, 140, 0), 3)  # BGR Blue
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)      # BGR Red pupil dot

    cv2.imwrite("scratch/test_detected_eyes.jpg", annotated)
    print("Saved test annotated image to scratch/test_detected_eyes.jpg")


if __name__ == "__main__":
    test_eye_detection("datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg")
