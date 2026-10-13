import io
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
backend_root = os.path.join(ROOT, "backend")
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.main import app
from app.models.user_model import User, AccessLog

client = TestClient(app)


def _make_jpeg_bytes(color=(10, 20, 30)):
    img = Image.new("RGB", (320, 240), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# -----------------------------------------------------------------------------
# 1. Diagnostic Mode Safety
# -----------------------------------------------------------------------------
def test_diagnostic_mode_sends_zero_hardware_commands_and_no_db_mutation():
    """
    Diagnostic mode (dry_run=True) must never send 'A' or 'U' to ESP32
    and must set hardware_action to 'SUPPRESSED_DIAGNOSTIC'.
    """
    payload = {"action": "CHECK-IN", "dry_run": "true"}
    files = [("file", ("face.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "final_decision": "VERIFIED",
            "failure_stage": None,
            "face_stage": {"matched": True, "candidate_user_id": 9, "candidate_user_name": "adiveppa"},
            "iris_stage": {"matched": True, "score": 0.85},
            "check_in_time": datetime.now().isoformat(),
            "check_in_time_formatted": "10:00:00 AM",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC"
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
            resp = client.post("/user/verify_identity", data=payload, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "VERIFIED"
    assert data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
    mock_esp.assert_not_called()


# -----------------------------------------------------------------------------
# 2. Live Verification Hardware Dispatch
# -----------------------------------------------------------------------------
def test_live_verified_dispatches_A_once():
    """
    Live mode (dry_run=False) with final_decision='VERIFIED' must call send_to_esp32('A') exactly once.
    """
    payload = {"action": "CHECK-IN", "dry_run": "false"}
    files = [("file", ("face.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "final_decision": "VERIFIED",
            "failure_stage": None,
            "face_stage": {"matched": True, "candidate_user_id": 9, "candidate_user_name": "adiveppa"},
            "iris_stage": {"matched": True, "score": 0.88},
            "hardware_action": "PENDING"
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
            mock_esp.return_value = {"status": "success", "command": "A", "ack": "[ESP32] AUTHORIZED"}
            resp = client.post("/user/verify_identity", data=payload, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "VERIFIED"
    assert data.get("hardware_action") == "SENT_A_AUTHORIZED"
    mock_esp.assert_called_once_with("A")


def test_live_biometric_mismatch_dispatches_U_once():
    """
    Live mode with failure_stage='BIOMETRIC_VERIFICATION' must call send_to_esp32('U') exactly once.
    """
    payload = {"action": "CHECK-IN", "dry_run": "false"}
    files = [("file", ("face.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "final_decision": "REJECTED",
            "failure_stage": "BIOMETRIC_VERIFICATION",
            "face_stage": {"matched": True, "candidate_user_id": 9, "candidate_user_name": "adiveppa"},
            "iris_stage": {"matched": False, "score": 0.35},
            "hardware_action": "PENDING"
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
            mock_esp.return_value = {"status": "success", "command": "U", "ack": "[ESP32] UNAUTHORIZED"}
            resp = client.post("/user/verify_identity", data=payload, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "REJECTED"
    assert data.get("failure_stage") == "BIOMETRIC_VERIFICATION"
    assert data.get("hardware_action") == "SENT_U_UNAUTHORIZED"
    mock_esp.assert_called_once_with("U")


# -----------------------------------------------------------------------------
# 3. Camera / Non-biometric Errors Must NOT Trigger Alarm
# -----------------------------------------------------------------------------
def test_identity_not_found_does_not_dispatch_U():
    """
    When face identity is not recognized (e.g. unknown face / camera error),
    send_to_esp32 must NOT be called.
    """
    payload = {"action": "CHECK-IN", "dry_run": "false"}
    files = [("file", ("face.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "face_stage": {"matched": False, "candidate_user_id": None},
            "iris_stage": {"executed": False},
            "hardware_action": "PENDING"
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
            resp = client.post("/user/verify_identity", data=payload, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "REJECTED"
    assert data.get("failure_stage") == "IDENTITY_IDENTIFICATION"
    assert "NO_COMMAND" in data.get("hardware_action", "")
    mock_esp.assert_not_called()


def test_access_control_rejection_does_not_dispatch_U():
    """
    When user is already inside (ACCESS_CONTROL failure), send_to_esp32 must NOT be called.
    """
    payload = {"action": "CHECK-IN", "dry_run": "false"}
    files = [("file", ("face.jpg", _make_jpeg_bytes(), "image/jpeg"))]

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "final_decision": "REJECTED",
            "failure_stage": "ACCESS_CONTROL",
            "decision_reason": "USER_ALREADY_INSIDE",
            "face_stage": {"matched": True, "candidate_user_id": 9, "candidate_user_name": "adiveppa"},
            "iris_stage": {"matched": True, "score": 0.88},
            "hardware_action": "PENDING"
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
            resp = client.post("/user/verify_identity", data=payload, files=files)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "REJECTED"
    assert data.get("failure_stage") == "ACCESS_CONTROL"
    assert data.get("hardware_action") == "NO_COMMAND_ACCESS_CONTROL"
    mock_esp.assert_not_called()


def test_empty_or_missing_file_does_not_dispatch_U():
    """
    Missing or empty file inputs must return error without dispatching hardware commands.
    """
    payload = {"action": "CHECK-IN", "dry_run": "false"}
    with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
        resp = client.post("/user/verify_identity", data=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert data.get("final_decision") == "REJECTED"
    assert data.get("hardware_action") == "NO_COMMAND_NO_FILE"
    mock_esp.assert_not_called()


# -----------------------------------------------------------------------------
# 4. Inside Block & Completed Visits Endpoints
# -----------------------------------------------------------------------------
def test_active_sessions_endpoint_returns_valid_structure():
    """
    /iris/active_sessions must return active_users list and total_inside count.
    """
    resp = client.get("/iris/active_sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "active_users" in data
    assert isinstance(data["active_users"], list)
    assert "total_inside" in data


def test_completed_visits_endpoint_returns_paired_histories():
    """
    /iris/completed_visits must return completed visits list and total count.
    """
    resp = client.get("/iris/completed_visits")
    assert resp.status_code == 200
    data = resp.json()
    assert "completed_visits" in data
    assert isinstance(data["completed_visits"], list)
    assert "total" in data


# -----------------------------------------------------------------------------
# 5. Manual Hardware Test Endpoint
# -----------------------------------------------------------------------------
def test_manual_hardware_test_endpoint_dispatches_valid_commands():
    """
    /iris/hardware/test must execute without mutating access records.
    """
    with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
        mock_esp.return_value = {"status": "success", "command": "A", "ack": "[ESP32] AUTHORIZED"}
        resp_a = client.post("/iris/hardware/test", data={"command": "A"})

    assert resp_a.status_code == 200
    data_a = resp_a.json()
    assert data_a.get("status") == "success"
    assert data_a.get("command") == "A"

    with patch("app.routers.iris_router.send_to_esp32") as mock_esp:
        mock_esp.return_value = {"status": "success", "command": "U", "ack": "[ESP32] UNAUTHORIZED"}
        resp_u = client.post("/iris/hardware/test", data={"command": "U"})

    assert resp_u.status_code == 200
    data_u = resp_u.json()
    assert data_u.get("status") == "success"
    assert data_u.get("command") == "U"
