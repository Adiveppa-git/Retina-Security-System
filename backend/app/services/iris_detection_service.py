import os
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from PIL import Image
from ai_models.utils.daugman_normalization import daugman_polar_normalization

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

LANDMARKER_TASK_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "ai_models/utils/face_landmarker.task")
)

_landmarker = None


def _ensure_landmarker_task():
    if not os.path.exists(LANDMARKER_TASK_PATH):
        os.makedirs(os.path.dirname(LANDMARKER_TASK_PATH), exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        try:
            print("[Iris Detection Service] Downloading MediaPipe face_landmarker.task...")
            urllib.request.urlretrieve(url, LANDMARKER_TASK_PATH)
            print("[Iris Detection Service] Downloaded face_landmarker.task successfully.")
        except Exception as e:
            print("[Iris Detection Service] Failed to download face_landmarker.task:", e)


def get_face_landmarker():
    global _landmarker
    if _landmarker is None:
        _ensure_landmarker_task()
        if os.path.exists(LANDMARKER_TASK_PATH):
            try:
                BaseOptions = mp.tasks.BaseOptions
                FaceLandmarker = mp.tasks.vision.FaceLandmarker
                FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
                VisionRunningMode = mp.tasks.vision.RunningMode

                options = FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=LANDMARKER_TASK_PATH),
                    running_mode=VisionRunningMode.IMAGE,
                    num_faces=1,
                    min_face_detection_confidence=0.3,
                    min_face_presence_confidence=0.3
                )
                _landmarker = FaceLandmarker.create_from_options(options)
            except Exception as e:
                print("[Iris Detection Service] Failed to instantiate MediaPipe FaceLandmarker:", e)
                _landmarker = None
    return _landmarker


class DetectionResult(tuple):
    """
    Backwards-compatible tuple subclass returning (polar_ribbon, annotated_image, face_det, left_det, right_det)
    while carrying structured payload dictionary via .payload attribute.
    """
    def __new__(cls, polar_ribbon, annotated_image, face_detected, left_eye_detected, right_eye_detected, payload=None):
        inst = super(DetectionResult, cls).__new__(cls, (polar_ribbon, annotated_image, face_detected, left_eye_detected, right_eye_detected))
        inst.payload = payload or {}
        return inst


def evaluate_eye_roi_quality(roi_gray, cx_roi, cy_roi, radius):
    """
    Calculates empirical eye ROI & pupil quality metrics:
    - sharpness (Laplacian variance)
    - contrast (std dev)
    - brightness (mean)
    - iris diameter in pixels
    - circularity & center validity
    """
    if roi_gray is None or roi_gray.size == 0:
        return {"valid": False, "reason": "EMPTY_ROI", "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "diameter_px": 0.0}

    h, w = roi_gray.shape[:2]
    sharpness = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())
    brightness = float(np.mean(roi_gray))
    contrast = float(np.std(roi_gray))
    diameter_px = float(2 * radius)

    # Check center location inside ROI
    if not (0 <= cx_roi < w and 0 <= cy_roi < h):
        return {"valid": False, "reason": "CENTER_OUTSIDE_ROI", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    # Eyelid closure check
    if h < 8 or (w / float(h)) > 4.5:
        return {"valid": False, "reason": "EYE_CLOSED_OR_OCCLUDED", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    # Brightness bounds (not pitch black / glare)
    if brightness < 12.0 or brightness > 245.0:
        return {"valid": False, "reason": "EXTREME_LIGHTING", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    # Contrast bounds
    if contrast < 6.0:
        return {"valid": False, "reason": "LOW_CONTRAST", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    # Sharpness / Blur check
    if sharpness < 4.0:
        return {"valid": False, "reason": "BLURRY_FRAME", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    # Resolution check
    if diameter_px < 12.0 or w < 25 or h < 15:
        return {"valid": False, "reason": "LOW_IRIS_RESOLUTION", "sharpness": round(sharpness, 2), "brightness": round(brightness, 2), "contrast": round(contrast, 2), "diameter_px": round(diameter_px, 2)}

    return {
        "valid": True,
        "reason": "OK",
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "diameter_px": round(diameter_px, 2)
    }


def detect_and_crop_iris(image_input):
    """
    Phase 3 Landmark-Based Precision Face, Dual-Eye & Single-Eye Detector.
    
    Pipeline:
    WEBCAM FRAME -> FACE LANDMARKS (MediaPipe 468/478) -> LEFT/RIGHT EYE ROI -> PUPIL/IRIS LOCALIZATION -> QUALITY GATE -> VALID CROP
    """
    annotated = None
    try:
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            np_img = np.array(pil_img)
        elif isinstance(image_input, np.ndarray):
            np_img = image_input.copy()
            if len(np_img.shape) == 2:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_GRAY2RGB)
        else:
            np_img = cv2.imread(image_input)
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

        h, w = np_img.shape[:2]
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        annotated = np_img.copy()

        landmarker = get_face_landmarker()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np_img)
        res = landmarker.detect(mp_image) if landmarker else None

        face_detected = False
        left_eye_detected = False
        left_iris_detected = False
        right_eye_detected = False
        right_iris_detected = False

        left_eye_bbox = [0, 0, 0, 0]
        right_eye_bbox = [0, 0, 0, 0]
        left_pupil_center = None
        right_pupil_center = None
        left_iris_radius = 0.0
        right_iris_radius = 0.0

        left_quality = {"valid": False, "reason": "NOT_DETECTED", "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "diameter_px": 0.0}
        right_quality = {"valid": False, "reason": "NOT_DETECTED", "sharpness": 0.0, "brightness": 0.0, "contrast": 0.0, "diameter_px": 0.0}
        failure_reason = "NO_FACE_DETECTED"

        if res and res.face_landmarks and len(res.face_landmarks) > 0:
            face_detected = True
            lm = res.face_landmarks[0]

            # Face Bounding Box from all landmarks
            fxs = [int(p.x * w) for p in lm]
            fys = [int(p.y * h) for p in lm]
            fx1, fy1, fx2, fy2 = max(0, min(fxs)), max(0, min(fys)), min(w, max(fxs)), min(h, max(fys))

            # Draw GREEN Face Box & Label
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            cv2.putText(annotated, "FACE DETECTED", (fx1 + 4, max(20, fy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Left Eye Landmark Cluster
            left_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            left_pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in left_indices])
            lx, ly, lw, lh = cv2.boundingRect(left_pts)
            left_eye_detected = True

            # Add 25% safety margin around eye landmarks for eye ROI
            pad_w = int(lw * 0.25)
            pad_h = int(lh * 0.35)
            lx1, ly1 = max(0, lx - pad_w), max(0, ly - pad_h)
            lx2, ly2 = min(w, lx + lw + pad_w), min(h, ly + lh + pad_h)
            left_eye_bbox = [lx1, ly1, lx2, ly2]

            # Right Eye Landmark Cluster
            right_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            right_pts = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in right_indices])
            rx, ry, rw, rh = cv2.boundingRect(right_pts)
            right_eye_detected = True

            pad_rw = int(rw * 0.25)
            pad_rh = int(rh * 0.35)
            rx1, ry1 = max(0, rx - pad_rw), max(0, ry - pad_rh)
            rx2, ry2 = min(w, rx + rw + pad_rw), min(h, ry + rh + pad_rh)
            right_eye_bbox = [rx1, ry1, rx2, ry2]

            # NO RECTANGLES DRAWN AROUND EYES!

            # Left Iris Location & Radius
            if len(lm) >= 478:
                left_cx = int(lm[468].x * w)
                left_cy = int(lm[468].y * h)
                l_edge = int(lm[469].x * w)
                left_r = max(6, int(abs(l_edge - left_cx)))
            else:
                left_cx = lx + lw // 2
                left_cy = ly + lh // 2
                left_r = max(6, min(lw, lh) // 2)

            # Left Eye Quality Validation
            left_roi_gray = gray[ly1:ly2, lx1:lx2]
            left_quality = evaluate_eye_roi_quality(left_roi_gray, left_cx - lx1, left_cy - ly1, left_r)

            left_eye_radius = max(14, int(lw * 0.8))
            if left_quality["valid"]:
                left_iris_detected = True
                left_pupil_center = [left_cx, left_cy]
                left_iris_radius = float(left_r)

                # Circular Orange/Gold Eye Ring (RGB: R=245, G=158, B=11)
                cv2.circle(annotated, (left_cx, left_cy), left_eye_radius, (245, 158, 11), 2)
                # Red Pupil Center Dot (RGB: R=255, G=0, B=0)
                cv2.circle(annotated, (left_cx, left_cy), 3, (255, 0, 0), -1)
                # Left Eye Label
                cv2.putText(annotated, "LEFT EYE", (max(0, left_cx - 28), max(16, left_cy - left_eye_radius - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 2)
            else:
                left_iris_detected = False
                left_pupil_center = None
                left_iris_radius = 0.0
                cv2.putText(annotated, "LEFT EYE: INVALID", (max(0, left_cx - 35), max(16, left_cy - left_eye_radius - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 2)

            # Right Iris Location & Radius
            if len(lm) >= 478:
                right_cx = int(lm[473].x * w)
                right_cy = int(lm[473].y * h)
                r_edge = int(lm[474].x * w)
                right_r = max(6, int(abs(r_edge - right_cx)))
            else:
                right_cx = rx + rw // 2
                right_cy = ry + rh // 2
                right_r = max(6, min(rw, rh) // 2)

            # Right Eye Quality Validation
            right_roi_gray = gray[ry1:ry2, rx1:rx2]
            right_quality = evaluate_eye_roi_quality(right_roi_gray, right_cx - rx1, right_cy - ry1, right_r)

            right_eye_radius = max(14, int(rw * 0.8))
            if right_quality["valid"]:
                right_iris_detected = True
                right_pupil_center = [right_cx, right_cy]
                right_iris_radius = float(right_r)

                # Circular Orange/Gold Eye Ring (RGB: R=245, G=158, B=11)
                cv2.circle(annotated, (right_cx, right_cy), right_eye_radius, (245, 158, 11), 2)
                # Red Pupil Center Dot (RGB: R=255, G=0, B=0)
                cv2.circle(annotated, (right_cx, right_cy), 3, (255, 0, 0), -1)
                # Right Eye Label
                cv2.putText(annotated, "RIGHT EYE", (max(0, right_cx - 30), max(16, right_cy - right_eye_radius - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 2)
            else:
                right_iris_detected = False
                right_pupil_center = None
                right_iris_radius = 0.0
                cv2.putText(annotated, "RIGHT EYE: INVALID", (max(0, right_cx - 35), max(16, right_cy - right_eye_radius - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 0), 2)

            if left_iris_detected or right_iris_detected:
                failure_reason = "NONE"
            else:
                failure_reason = left_quality.get("reason", "QUALITY_GATE_FAILED")

        else:
            # Fallback ONLY for single close-up dataset eye images (e.g. CASIA 320x280 or small cropped single-eye templates)
            if (w < 450 and h < 450) or (abs(w - h) < 120 and (w * h) < 180000):
                blurred = cv2.GaussianBlur(gray, (7, 7), 0)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)

                if min_val < 70 and (max_val - min_val) > 60:
                    cx_single, cy_single = min_loc
                    r_single = min(h, w) // 4
                    single_quality = evaluate_eye_roi_quality(gray, cx_single, cy_single, r_single)

                    if single_quality["valid"]:
                        face_detected = True
                        left_eye_detected = True
                        left_iris_detected = True
                        right_eye_detected = True
                        right_iris_detected = True
                        left_pupil_center = [cx_single, cy_single]
                        left_iris_radius = float(r_single)
                        right_pupil_center = [cx_single, cy_single]
                        right_iris_radius = float(r_single)
                        left_quality = single_quality
                        right_quality = single_quality

                        # Compute real, valid, clamped eye bounding boxes based on detected eye center and radius
                        pad_w = max(16, int(r_single * 1.8))
                        pad_h = max(12, int(r_single * 1.3))
                        fb_x1 = max(0, cx_single - pad_w)
                        fb_y1 = max(0, cy_single - pad_h)
                        fb_x2 = min(w, cx_single + pad_w)
                        fb_y2 = min(h, cy_single + pad_h)

                        if (fb_x2 - fb_x1) >= 20 and (fb_y2 - fb_y1) >= 12:
                            left_eye_bbox = [fb_x1, fb_y1, fb_x2, fb_y2]
                            right_eye_bbox = [fb_x1, fb_y1, fb_x2, fb_y2]
                            failure_reason = "NONE"
                        else:
                            left_iris_detected = False
                            right_iris_detected = False
                            failure_reason = "FALLBACK_BOUNDING_BOX_INVALID"

                        cv2.circle(annotated, (cx_single, cy_single), r_single, (255, 0, 0), 3)
                        cv2.circle(annotated, (cx_single, cy_single), 3, (255, 0, 0), -1)
                    else:
                        cv2.putText(annotated, "EYE NOT DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        failure_reason = single_quality.get("reason", "SINGLE_EYE_QUALITY_FAILED")
                else:
                    cv2.putText(annotated, "FACE NOT DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    failure_reason = "NO_FACE_DETECTED"
            else:
                cv2.putText(annotated, "FACE NOT DETECTED", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                failure_reason = "NO_FACE_DETECTED"

        # Extract primary valid eye crop for Daugman polar ribbon
        polar_ribbon = None
        if left_iris_detected and left_pupil_center is not None and left_iris_radius > 0:
            cx, cy, r_val = left_pupil_center[0], left_pupil_center[1], int(left_iris_radius)
        elif right_iris_detected and right_pupil_center is not None and right_iris_radius > 0:
            cx, cy, r_val = right_pupil_center[0], right_pupil_center[1], int(right_iris_radius)
        else:
            cx, cy, r_val = None, None, None

        if cx is not None and cy is not None and r_val is not None:
            x1 = max(0, cx - r_val - 10)
            y1 = max(0, cy - r_val - 10)
            x2 = min(w, cx + r_val + 10)
            y2 = min(h, cy + r_val + 10)

            if (x2 - x1) >= 12 and (y2 - y1) >= 12:
                cropped_eye = np_img[y1:y2, x1:x2]
                if cropped_eye.size > 0:
                    polar_ribbon = daugman_polar_normalization(Image.fromarray(cropped_eye))

        payload = {
            "face_detected": face_detected,
            "left_eye_detected": left_eye_detected,
            "left_iris_detected": left_iris_detected,
            "right_eye_detected": right_eye_detected,
            "right_iris_detected": right_iris_detected,
            "left_eye_bbox": left_eye_bbox,
            "right_eye_bbox": right_eye_bbox,
            "left_pupil_center": left_pupil_center,
            "right_pupil_center": right_pupil_center,
            "left_iris_radius": left_iris_radius,
            "right_iris_radius": right_iris_radius,
            "left_iris_quality": left_quality,
            "right_iris_quality": right_quality,
            "failure_reason": failure_reason
        }

        return DetectionResult(polar_ribbon, Image.fromarray(annotated), face_detected, left_iris_detected, right_iris_detected, payload=payload)

    except Exception as e:
        print("[Iris Detection Exception]:", e)
        fallback_pil = None
        try:
            if isinstance(image_input, Image.Image):
                fallback_pil = image_input.convert("RGB")
            elif isinstance(image_input, np.ndarray) and image_input.size > 0:
                fallback_pil = Image.fromarray(image_input).convert("RGB")
            elif isinstance(image_input, str) and os.path.exists(image_input):
                fallback_pil = Image.open(image_input).convert("RGB")
        except Exception:
            fallback_pil = None

        if fallback_pil is None:
            fallback_pil = Image.new("RGB", (100, 100), (0, 0, 0))

        ann_pil = Image.fromarray(annotated) if (annotated is not None and isinstance(annotated, np.ndarray) and annotated.size > 0) else fallback_pil
        payload = {
            "face_detected": False,
            "left_eye_detected": False,
            "left_iris_detected": False,
            "right_eye_detected": False,
            "right_iris_detected": False,
            "left_eye_bbox": [0, 0, 0, 0],
            "right_eye_bbox": [0, 0, 0, 0],
            "left_pupil_center": None,
            "right_pupil_center": None,
            "left_iris_radius": 0.0,
            "right_iris_radius": 0.0,
            "failure_reason": str(e)
        }
        return DetectionResult(None, ann_pil, False, False, False, payload=payload)


def detect_iris(image):
    return detect_and_crop_iris(image)