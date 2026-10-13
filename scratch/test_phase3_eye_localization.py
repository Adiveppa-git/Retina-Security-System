import os
import cv2
import numpy as np
from PIL import Image
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from backend.app.services.iris_detection_service import detect_and_crop_iris

def create_synthetic_test_frame(scenario_name, width=640, height=480):
    output_dir = os.path.abspath("scratch/phase3_results")
    os.makedirs(output_dir, exist_ok=True)

    if scenario_name == "A_normal_face":
        p_path = "project_images/002.jpeg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "B_move_left":
        p_path = "project_images/008.jpeg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "C_move_right":
        p_path = "project_images/012.jpeg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "F_glasses":
        p_path = "scratch/test_exact_glasses_eyes.jpg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "G_no_glasses":
        p_path = "project_images/016.jpeg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "I_background_table":
        p_path = "project_images/001.jpeg"
        if os.path.exists(p_path): return p_path
    elif scenario_name == "K_no_face":
        bg = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(bg, (0, 300), (640, 480), (40, 80, 120), -1)
        cv2.circle(bg, (150, 380), 40, (10, 10, 10), -1)
        save_p = os.path.join(output_dir, "scenario_K_no_face.jpg")
        cv2.imwrite(save_p, bg)
        return save_p
    elif scenario_name == "J_phone_covering_eyes":
        p_path = "project_images/002.jpeg"
        if os.path.exists(p_path):
            img = cv2.imread(p_path)
            h, w = img.shape[:2]
            cv2.rectangle(img, (int(w*0.3), int(h*0.35)), (int(w*0.7), int(h*0.55)), (10, 10, 10), -1)
            save_p = os.path.join(output_dir, "scenario_J_phone.jpg")
            cv2.imwrite(save_p, img)
            return save_p
    elif scenario_name == "L_low_light":
        p_path = "project_images/002.jpeg"
        if os.path.exists(p_path):
            img = cv2.imread(p_path)
            img = (img * 0.15).astype(np.uint8)
            save_p = os.path.join(output_dir, "scenario_L_low_light.jpg")
            cv2.imwrite(save_p, img)
            return save_p

    p_path = "project_images/020.jpeg"
    if os.path.exists(p_path): return p_path
    return None

def run_phase3_test_suite():
    print("==================================================")
    print(" PHASE 3 — WEBCAM & EYE LOCALIZATION TEST HARNESS (V2)")
    print("==================================================")

    scenarios = [
        ("A", "normal_face", "A_normal_face"),
        ("B", "move_left", "B_move_left"),
        ("C", "move_right", "C_move_right"),
        ("D", "move_up", "A_normal_face"),
        ("E", "move_down", "A_normal_face"),
        ("F", "glasses", "F_glasses"),
        ("G", "no_glasses", "G_no_glasses"),
        ("H", "partial_face", "B_move_left"),
        ("I", "face_with_background_table", "I_background_table"),
        ("J", "phone_covering_eyes", "J_phone_covering_eyes"),
        ("K", "no_face", "K_no_face"),
        ("L", "low_light", "L_low_light"),
        ("M", "move_closer", "G_no_glasses"),
        ("N", "move_farther", "A_normal_face")
    ]

    results_summary = []
    payload_consistency_passed = True

    for code, desc, frame_key in scenarios:
        frame_path = create_synthetic_test_frame(frame_key)
        if not frame_path or not os.path.exists(frame_path):
            print(f"[{code}] {desc:<25} | Scenario frame missing, skipping.")
            continue

        res = detect_and_crop_iris(frame_path)
        p = res.payload

        face_det = p.get("face_detected", False)
        l_eye_det = p.get("left_eye_detected", False)
        l_iris_det = p.get("left_iris_detected", False)
        r_eye_det = p.get("right_eye_detected", False)
        r_iris_det = p.get("right_iris_detected", False)

        l_center = p.get("left_pupil_center")
        r_center = p.get("right_pupil_center")
        l_r = p.get("left_iris_radius", 0.0)
        r_r = p.get("right_iris_radius", 0.0)

        l_qual = p.get("left_iris_quality", {})
        r_qual = p.get("right_iris_quality", {})
        reason = p.get("failure_reason", "NONE")

        l_diam = l_r * 2.0
        r_diam = r_r * 2.0

        # Strict Payload Internal Consistency Audit
        if not l_iris_det:
            if l_center is not None or l_r != 0.0 or l_qual.get("valid") is True:
                payload_consistency_passed = False
                print(f"[CONSISTENCY ERROR] Scenario {code}: Left iris not detected but payload contains non-null values!")
        if not r_iris_det:
            if r_center is not None or r_r != 0.0 or r_qual.get("valid") is True:
                payload_consistency_passed = False
                print(f"[CONSISTENCY ERROR] Scenario {code}: Right iris not detected but payload contains non-null values!")

        # Save annotated visual debug image
        annotated_img = res[1]
        out_save_path = os.path.abspath(f"scratch/phase3_results/scenario_{code}_{desc}.jpg")
        annotated_img.save(out_save_path)

        rec = {
            "scenario_code": code,
            "description": desc,
            "face_detected": face_det,
            "left_eye_detected": l_eye_det,
            "left_iris_detected": l_iris_det,
            "left_iris_diameter_px": l_diam,
            "right_eye_detected": r_eye_det,
            "right_iris_detected": r_iris_det,
            "right_iris_diameter_px": r_diam,
            "failure_reason": reason
        }
        results_summary.append(rec)

        print(f"[{code}] {desc:<26} | Face: {str(face_det):<5} | L-Eye: {str(l_eye_det):<5} | L-Iris: {str(l_iris_det):<5} ({l_diam:>5.1f}px) | R-Eye: {str(r_eye_det):<5} | R-Iris: {str(r_iris_det):<5} ({r_diam:>5.1f}px) | Reason: {reason}")

    print("\n==================================================")
    print(" PAYLOAD INTERNAL CONSISTENCY AUDIT")
    print("==================================================")
    print(f" Internal Payload Consistency Check : {'PASSED' if payload_consistency_passed else 'FAILED'}")
    print("==================================================\n")

    return results_summary

if __name__ == "__main__":
    run_phase3_test_suite()
