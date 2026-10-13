import urllib.request
import json
import os
import mimetypes
import uuid
import sys

workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, workspace)
sys.path.insert(0, os.path.join(workspace, "backend"))
from app.services.iris_detection_service import detect_and_crop_iris
from PIL import Image
import numpy as np

def post_multipart(url, fields, files):
    boundary = uuid.uuid4().hex
    body = []
    for k, v in fields.items():
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode('utf-8'))
    for k, path in files.items():
        bname = os.path.basename(path)
        mime = mimetypes.guess_type(path)[0] or 'application/octet-stream'
        with open(path, 'rb') as f:
            data = f.read()
        body.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"; filename="{bname}"\r\nContent-Type: {mime}\r\n\r\n'.encode('utf-8'))
        body.append(data + b'\r\n')
    body.append(f'--{boundary}--\r\n'.encode('utf-8'))
    req = urllib.request.Request(url, data=b''.join(body))
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req) as res:
        return json.loads(res.read().decode())

def run_tests():
    print("=" * 80)
    print("REAL PRODUCTION END-TO-END VERIFICATION TEST (LIVE BACKEND + REAL DB + ESP32)")
    print("=" * 80)

    # 1. Genuine Live Verification Attempts
    print("\n--- STEP 4: GENUINE LIVE ATTEMPTS (ENROLLED USER #6 ADIVEPPA) ---")
    gen_samples = [
        os.path.join(workspace, "scratch", "live_sample_1.jpg"),
        os.path.join(workspace, "scratch", "live_sample_3.jpg"),
        os.path.join(workspace, "scratch", "live_sample_4.jpg"),
        os.path.join(workspace, "scratch", "live_sample_8.jpg")
    ]

    for p in gen_samples:
        bname = os.path.basename(p)
        # Inspect local detection metrics directly
        img_np = np.array(Image.open(p).convert("RGB"))
        h, w = img_np.shape[:2]
        det = detect_and_crop_iris(img_np).payload

        # Call live production API
        out = post_multipart('http://127.0.0.1:8000/iris/verify_1to1', {'claimed_user_id': 6, 'action': 'CHECK-IN'}, {'file': p})

        l_r = det.get("left_iris_radius", 0.0)
        r_r = det.get("right_iris_radius", 0.0)
        l_diam = 2 * l_r
        r_diam = 2 * r_r
        l_q = det.get("left_iris_quality", {})
        r_q = det.get("right_iris_quality", {})

        print(f"\nAttempt: {bname}")
        print(f"  Frame Resolution      : {w}x{h}")
        print(f"  Detected Eyes         : L={det.get('left_iris_detected')}, R={det.get('right_iris_detected')}")
        print(f"  Iris Diameters        : Left={l_diam:.1f}px, Right={r_diam:.1f}px")
        print(f"  Quality Results       : Left={l_q.get('reason')}, Right={r_q.get('reason')}")
        print(f"  Similarity Score      : {out.get('best_similarity'):.4f}")
        print(f"  Threshold Used        : {out.get('diagnostic_threshold')}")
        print(f"  Final Decision        : {out.get('final_diagnostic_status')} ({out.get('decision_reason')})")
        print(f"  Biometric Verified    : {out.get('biometric_verified')}")

    # 2. Impostor Verification Attempts
    print("\n--- STEP 5: IMPOSTOR ATTEMPTS (UNAUTHORIZED PERSONS CLAIMING USER #6) ---")
    imp_samples = [
        ("Renkigouda", os.path.join(workspace, "project_images", "002.jpeg")),
        ("Renkigouda (Alt)", os.path.join(workspace, "project_images", "020.jpeg")),
        ("Unknown Subject 1", os.path.join(workspace, "project_images", "012.jpeg")),
        ("Unknown Subject 2", os.path.join(workspace, "project_images", "016.jpeg"))
    ]

    for label, p in imp_samples:
        bname = os.path.basename(p)
        img_np = np.array(Image.open(p).convert("RGB"))
        h, w = img_np.shape[:2]
        det = detect_and_crop_iris(img_np).payload

        # Call live production API
        out = post_multipart('http://127.0.0.1:8000/iris/verify_1to1', {'claimed_user_id': 6, 'action': 'CHECK-IN'}, {'file': p})

        l_r = det.get("left_iris_radius", 0.0)
        r_r = det.get("right_iris_radius", 0.0)
        l_diam = 2 * l_r
        r_diam = 2 * r_r
        l_q = det.get("left_iris_quality", {})
        r_q = det.get("right_iris_quality", {})

        print(f"\nAttempt: {label} ({bname})")
        print(f"  Frame Resolution      : {w}x{h}")
        print(f"  Detected Eyes         : L={det.get('left_iris_detected')}, R={det.get('right_iris_detected')}")
        print(f"  Iris Diameters        : Left={l_diam:.1f}px, Right={r_diam:.1f}px")
        print(f"  Quality Results       : Left={l_q.get('reason')}, Right={r_q.get('reason')}")
        print(f"  Similarity Score      : {out.get('best_similarity'):.4f}")
        print(f"  Threshold Used        : {out.get('diagnostic_threshold')}")
        print(f"  Final Decision        : {out.get('final_diagnostic_status')} ({out.get('decision_reason')})")
        print(f"  Biometric Verified    : {out.get('biometric_verified')}")

if __name__ == "__main__":
    run_tests()
