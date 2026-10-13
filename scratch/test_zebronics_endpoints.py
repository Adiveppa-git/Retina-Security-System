import os
import sys
import io
import time
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_api_endpoints():
    print("==================================================")
    print(" TESTING ZEBRONICS FASTAPI ENDPOINTS")
    print("==================================================")

    # 1. Test Camera Status Endpoint
    print("\n--- 1. Testing GET /iris/camera/status ---")
    resp_status = client.get("/iris/camera/status")
    print("Status code:", resp_status.status_code)
    data_status = resp_status.json()
    print("Response:", data_status)
    assert resp_status.status_code == 200
    assert data_status["camera_source"] == "Zebronics USB Webcam"
    assert data_status["camera_online"] is True
    assert data_status["metadata"]["backend"] in ("DSHOW", "MSMF")
    assert data_status["metadata"]["mean_brightness"] > 30.0
    print("[PASS] Camera status endpoint verified successfully!")

    # 2. Test POST /iris/verify
    print("\n--- 2. Testing POST /iris/verify ---")
    img = Image.new("RGB", (640, 480), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    resp_verify = client.post("/iris/verify", files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")})
    print("Status code:", resp_verify.status_code)
    data_verify = resp_verify.json()
    print("camera_source in response:", data_verify.get("camera_source"))
    assert data_verify.get("camera_source") == "Zebronics USB Webcam"
    print("[PASS] /iris/verify camera_source verified!")

    # 3. Test POST /iris/verify_1to1
    print("\n--- 3. Testing POST /iris/verify_1to1 ---")
    buf.seek(0)
    resp_1to1 = client.post(
        "/iris/verify_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"},
        files={"file": ("test.jpg", buf.getvalue(), "image/jpeg")}
    )
    data_1to1 = resp_1to1.json()
    print("camera_source in response:", data_1to1.get("camera_source"))
    assert data_1to1.get("camera_source") == "Zebronics USB Webcam (Single-Frame 1:1)"
    print("[PASS] /iris/verify_1to1 camera_source verified!")

    # 4. Test POST /iris/verify_sequence_1to1
    print("\n--- 4. Testing POST /iris/verify_sequence_1to1 ---")
    seq_files = [("files", (f"seq_{i}.jpg", buf.getvalue(), "image/jpeg")) for i in range(3)]
    resp_seq = client.post(
        "/iris/verify_sequence_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"},
        files=seq_files
    )
    data_seq = resp_seq.json()
    print("camera_source in response:", data_seq.get("camera_source"))
    assert data_seq.get("camera_source") == "Zebronics USB Webcam (5-7 Frame 1:1 Sequence)"
    print("[PASS] /iris/verify_sequence_1to1 camera_source verified!")

    # 5. Test POST /iris/verify_camera_1to1 (Live Zebronics Capture -> Biometric Pipeline -> ESP32)
    print("\n--- 5. Testing POST /iris/verify_camera_1to1 (Live Zebronics Hardware Capture) ---")
    t0 = time.time()
    resp_live = client.post(
        "/iris/verify_camera_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"}
    )
    duration = time.time() - t0
    print(f"Status code: {resp_live.status_code} (took {duration:.2f}s)")
    data_live = resp_live.json()
    print("camera_source:", data_live.get("camera_source"))
    print("final_diagnostic_status:", data_live.get("final_diagnostic_status"))
    print("camera_metadata:", data_live.get("camera_metadata"))
    assert resp_live.status_code == 200
    assert "Zebronics USB Webcam" in data_live.get("camera_source")
    print("[PASS] Live Zebronics camera acquisition & 1:1 verification pipeline verified!")

    print("\n==================================================")
    print(" ALL ZEBRONICS FASTAPI ENDPOINT TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_api_endpoints()
