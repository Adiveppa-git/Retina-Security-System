import io
import os
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
backend_root = os.path.join(ROOT, "backend")
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.main import app

client = TestClient(app)


def _make_jpeg_bytes(color=(10, 20, 30)):
    img = Image.new("RGB", (320, 240), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detection_failure_does_not_trigger_unauthorized_hardware_signal():
    payload = {"claimed_user_id": 1, "action": "CHECK-IN", "dry_run": False}
    files = [("file", ("frame.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.iris_router.recognize_iris_v2_1to1_diagnostic") as mock_recognize:
        mock_recognize.return_value = {
            "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
            "decision_reason": "NO_FACE_DETECTED",
            "biometric_verified": False,
            "claimed_user_id": 1,
            "claimed_user_name": "Test User",
            "message": "1:1 Verification: Position face clearly in front of camera!",
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            response = client.post("/iris/verify_1to1", data=payload, files=files)

    assert response.status_code == 200, response.text
    status = response.json().get("final_diagnostic_status")
    assert status == "CLAIMED_IDENTITY_NOT_VERIFIED"
    mock_send.assert_not_called()
    assert response.json().get("hardware_action") in {"NO_COMMAND_AUTHENTICATION_FAILED", "SUPPRESSED_DIAGNOSTIC"}


def test_system_error_does_not_trigger_unauthorized_hardware_signal():
    payload = {"claimed_user_id": 1, "action": "CHECK-IN", "dry_run": False}
    files = [("file", ("frame.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.iris_router.recognize_iris_v2_1to1_diagnostic") as mock_recognize:
        mock_recognize.return_value = {
            "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
            "decision_reason": "ERROR: camera unavailable",
            "biometric_verified": False,
            "claimed_user_id": 1,
            "claimed_user_name": "Test User",
            "message": "camera unavailable",
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            response = client.post("/iris/verify_1to1", data=payload, files=files)

    assert response.status_code == 200, response.text
    assert response.json().get("final_diagnostic_status") == "CLAIMED_IDENTITY_NOT_VERIFIED"
    mock_send.assert_not_called()
    assert response.json().get("hardware_action") in {"NO_COMMAND_SYSTEM_ERROR", "NO_COMMAND_AUTHENTICATION_FAILED"}
