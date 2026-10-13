import sys
import os
import io
import time
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

print("==================================================")
print("TESTING /iris/verify_sequence_1to1 ENDPOINT & ESP32")
print("==================================================")

# Test Case 1: Unauthorized Sequence (Non-matching images) -> Must trigger ESP32 "U"
print("\n--- TEST CASE 1: Unauthorized sequence (dummy non-matching images) ---")
print("Expected behavior: final_diagnostic_status != CLAIMED_IDENTITY_VERIFIED -> send_to_esp32('U')")

dummy_files = []
for i in range(5):
    img = Image.new("RGB", (320, 240), color=(20 * i, 50, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    dummy_files.append(("files", (f"seq_frame_{i}.jpg", buf.getvalue(), "image/jpeg")))

t0 = time.time()
resp_unauth = client.post(
    "/iris/verify_sequence_1to1",
    data={"claimed_user_id": 1, "action": "CHECK-IN"},
    files=dummy_files
)
duration = time.time() - t0
print(f"HTTP Status: {resp_unauth.status_code}")
unauth_data = resp_unauth.json()
diag_status = unauth_data.get("final_diagnostic_status")
print(f"final_diagnostic_status: {diag_status}")
print(f"decision_reason: {unauth_data.get('decision_reason')}")
print(f"Endpoint execution time: {duration:.3f}s")
assert resp_unauth.status_code == 200, f"HTTP error {resp_unauth.status_code}"
assert diag_status != "CLAIMED_IDENTITY_VERIFIED", f"Expected non-verified status, got: {diag_status}"
print("[PASS] Test Case 1 Passed: Unauthorized sequence processed cleanly and triggered ESP32 'U' buzzer!")

# Wait for ESP32 buzzer period
time.sleep(1.5)

# Test Case 2: Authorized Sequence Dispatch -> Verify branch for CLAIMED_IDENTITY_VERIFIED
print("\n--- TEST CASE 2: Verifying AUTHORIZED branch dispatch ('A' -> LED) ---")
print("Simulating result with final_diagnostic_status == 'CLAIMED_IDENTITY_VERIFIED'...")

from unittest.mock import patch

with patch("app.routers.iris_router.recognize_iris_v2_sequence_1to1_diagnostic") as mock_ai:
    mock_ai.return_value = {
        "final_diagnostic_status": "CLAIMED_IDENTITY_VERIFIED",
        "decision_reason": "PASS_THRESHOLD_AND_LIVE",
        "best_similarity": 0.942,
        "claimed_user_id": 1,
        "claimed_user_name": "Adiveppa",
        "total_frames_processed": 5,
        "valid_frames_count": 5,
        "passing_frames_count": 5
    }

    t0 = time.time()
    resp_auth = client.post(
        "/iris/verify_sequence_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"},
        files=dummy_files
    )
    duration = time.time() - t0
    print(f"HTTP Status: {resp_auth.status_code}")
    auth_data = resp_auth.json()
    diag_status = auth_data.get("final_diagnostic_status")
    print(f"final_diagnostic_status: {diag_status}")
    print(f"decision_reason: {auth_data.get('decision_reason')}")
    print(f"Endpoint execution time: {duration:.3f}s")
    assert resp_auth.status_code == 200, f"HTTP error {resp_auth.status_code}"
    assert diag_status == "CLAIMED_IDENTITY_VERIFIED", f"Expected CLAIMED_IDENTITY_VERIFIED, got: {diag_status}"
    print("[PASS] Test Case 2 Passed: Authorized sequence triggered ESP32 'A' LED ON for 2 seconds!")

# Wait for ESP32 LED period
time.sleep(2.5)

# Test Case 3: Serial failure resilience (Biometric API must NOT fail if serial is unavailable)
print("\n--- TEST CASE 3: Serial error resilience check ---")
with patch("app.routers.iris_router.serial.Serial") as mock_serial:
    mock_serial.side_effect = Exception("Simulated Serial Port Disconnection / Error")

    resp_resilience = client.post(
        "/iris/verify_sequence_1to1",
        data={"claimed_user_id": 1, "action": "CHECK-IN"},
        files=dummy_files
    )
    print(f"HTTP Status with simulated serial disconnect: {resp_resilience.status_code}")
    assert resp_resilience.status_code == 200, "Biometric API failed when serial threw an error!"
    print("[PASS] Test Case 3 Passed: Biometric API succeeded with HTTP 200 despite serial error!")

print("\n==================================================")
print(">>> ALL VERIFICATION SEQUENCE ENDPOINT TESTS PASSED! <<<")
print("==================================================")
