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


def _make_jpeg_bytes(color=(30, 40, 50)):
    img = Image.new("RGB", (320, 240), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_user_verify_identity_dry_run_true_suppresses_hardware():
    """Confirms dry_run='true' suppresses ESP32 serial commands on rejection."""
    jpeg_data = _make_jpeg_bytes()
    files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]
    data = {"action": "CHECK-IN", "dry_run": "true"}

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "BIOMETRIC_VERIFICATION",
            "face_stage": {"matched": True, "candidate_user_id": 1, "candidate_user_name": "Test User"},
            "iris_stage": {"matched": False, "score": 0.45}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            response = client.post("/user/verify_identity", data=data, files=files)

    assert response.status_code == 200, response.text
    res_data = response.json()
    assert res_data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
    mock_send.assert_not_called()


def test_user_verify_identity_dry_run_false_triggers_unauthorized():
    """Confirms dry_run='false' properly triggers send_to_esp32('U') on biometric rejection."""
    jpeg_data = _make_jpeg_bytes()
    files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]
    data = {"action": "CHECK-IN", "dry_run": "false"}

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "BIOMETRIC_VERIFICATION",
            "face_stage": {"matched": True, "candidate_user_id": 1, "candidate_user_name": "Test User"},
            "iris_stage": {"matched": False, "score": 0.45}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            mock_send.return_value = {"status": "success", "command": "U", "ack": "[ESP32] UNAUTHORIZED"}
            response = client.post("/user/verify_identity", data=data, files=files)

    assert response.status_code == 200, response.text
    res_data = response.json()
    assert res_data.get("hardware_action") == "SENT_U_UNAUTHORIZED"
    mock_send.assert_called_once_with("U")


def test_user_verify_identity_dry_run_false_triggers_authorized_pulse():
    """Confirms dry_run='false' triggers send_to_esp32('A') only on verified access."""
    jpeg_data = _make_jpeg_bytes()
    files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]
    data = {"action": "CHECK-IN", "dry_run": "false"}

    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "status": "success",
            "final_decision": "VERIFIED",
            "failure_stage": None,
            "face_stage": {"matched": True, "candidate_user_id": 1, "candidate_user_name": "Test User"},
            "iris_stage": {"matched": True, "score": 0.88}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            mock_send.return_value = {"status": "success", "command": "A", "ack": "[ESP32] AUTHORIZED"}
            response = client.post("/user/verify_identity", data=data, files=files)

    assert response.status_code == 200, response.text
    res_data = response.json()
    assert res_data.get("hardware_action") == "SENT_A_AUTHORIZED"
    mock_send.assert_called_once_with("A")


def test_iris_hardware_test_endpoint_commands():
    """Confirms POST /iris/hardware/test correctly accepts 'A' and 'U' and rejects invalid commands."""
    with patch("app.routers.iris_router.send_to_esp32") as mock_send:
        mock_send.return_value = {"status": "success", "command": "U", "ack": "ACK_U"}
        res_u = client.post("/iris/hardware/test", data={"command": "U"})
        assert res_u.status_code == 200
        assert res_u.json().get("status") == "success"
        assert res_u.json().get("hardware_action") == "TEST_U_EXECUTED"
        mock_send.assert_called_with("U")

    with patch("app.routers.iris_router.send_to_esp32") as mock_send:
        mock_send.return_value = {"status": "success", "command": "A", "ack": "ACK_A"}
        res_a = client.post("/iris/hardware/test", data={"command": "A"})
        assert res_a.status_code == 200
        assert res_a.json().get("status") == "success"
        assert res_a.json().get("hardware_action") == "TEST_A_EXECUTED"
        mock_send.assert_called_with("A")

    with patch("app.routers.iris_router.send_to_esp32") as mock_send:
        res_invalid = client.post("/iris/hardware/test", data={"command": "UNKNOWN"})
        assert res_invalid.status_code == 200
        assert res_invalid.json().get("status") == "error"
        assert res_invalid.json().get("error_code") == "INVALID_COMMAND"
        mock_send.assert_not_called()


def test_user_verify_identity_live_identity_identification_failure_dispatches_U():
    """Confirms live mode (dry_run='false') sends 'U' to ESP32 on definitive identity identification failure."""
    jpeg_data = _make_jpeg_bytes()
    data = {"action": "CHECK-IN", "dry_run": "false"}

    # 1. Single-frame mode
    files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]
    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "face_stage": {
                "matched": False,
                "status": "success",
                "reason": "IDENTITY_NOT_FOUND",
                "candidate_user_id": None,
                "candidate_user_name": None
            },
            "iris_stage": {"executed": False}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            mock_send.return_value = {"status": "success", "command": "U", "ack": "[ESP32] UNAUTHORIZED"}
            response = client.post("/user/verify_identity", data=data, files=files)

    assert response.status_code == 200, response.text
    res_data = response.json()
    assert res_data.get("final_decision") == "REJECTED"
    assert res_data.get("failure_stage") == "IDENTITY_IDENTIFICATION"
    assert res_data.get("hardware_action") == "SENT_U_UNAUTHORIZED"
    mock_send.assert_called_once_with("U")

    # 2. Multi-frame sequence mode (Verify Iris live capture workflow)
    multi_files = [
        ("files", ("f1.jpg", jpeg_data, "image/jpeg")),
        ("files", ("f2.jpg", jpeg_data, "image/jpeg"))
    ]
    with patch("app.routers.user_router.verify_two_stage_sequence_identity") as mock_seq_verify:
        mock_seq_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "face_stage": {
                "matched": False,
                "status": "success",
                "reason": "IDENTITY_NOT_FOUND",
                "candidate_user_id": None,
                "candidate_user_name": None
            },
            "iris_stage": {"executed": False}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send_seq:
            mock_send_seq.return_value = {"status": "success", "command": "U", "ack": "[ESP32] UNAUTHORIZED"}
            seq_resp = client.post("/user/verify_identity", data=data, files=multi_files)

    assert seq_resp.status_code == 200, seq_resp.text
    seq_data = seq_resp.json()
    assert seq_data.get("final_decision") == "REJECTED"
    assert seq_data.get("failure_stage") == "IDENTITY_IDENTIFICATION"
    assert seq_data.get("hardware_action") == "SENT_U_UNAUTHORIZED"
    mock_send_seq.assert_called_once_with("U")


def test_user_verify_identity_diagnostic_identity_identification_failure_suppresses_hardware():
    """Confirms diagnostic mode (dry_run='true') sends no hardware command on identity identification failure."""
    jpeg_data = _make_jpeg_bytes()
    data = {"action": "CHECK-IN", "dry_run": "true"}

    # 1. Single-frame mode
    files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]
    with patch("app.routers.user_router.verify_two_stage_identity") as mock_verify:
        mock_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "face_stage": {
                "matched": False,
                "status": "success",
                "reason": "IDENTITY_NOT_FOUND",
                "candidate_user_id": None,
                "candidate_user_name": None
            },
            "iris_stage": {"executed": False}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send:
            response = client.post("/user/verify_identity", data=data, files=files)

    assert response.status_code == 200, response.text
    res_data = response.json()
    assert res_data.get("final_decision") == "REJECTED"
    assert res_data.get("failure_stage") == "IDENTITY_IDENTIFICATION"
    assert res_data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
    mock_send.assert_not_called()

    # 2. Multi-frame sequence mode
    multi_files = [
        ("files", ("f1.jpg", jpeg_data, "image/jpeg")),
        ("files", ("f2.jpg", jpeg_data, "image/jpeg"))
    ]
    with patch("app.routers.user_router.verify_two_stage_sequence_identity") as mock_seq_verify:
        mock_seq_verify.return_value = {
            "status": "success",
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "face_stage": {
                "matched": False,
                "status": "success",
                "reason": "IDENTITY_NOT_FOUND",
                "candidate_user_id": None,
                "candidate_user_name": None
            },
            "iris_stage": {"executed": False}
        }
        with patch("app.routers.iris_router.send_to_esp32") as mock_send_seq:
            seq_resp = client.post("/user/verify_identity", data=data, files=multi_files)

    assert seq_resp.status_code == 200, seq_resp.text
    seq_data = seq_resp.json()
    assert seq_data.get("final_decision") == "REJECTED"
    assert seq_data.get("failure_stage") == "IDENTITY_IDENTIFICATION"
    assert seq_data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
    mock_send_seq.assert_not_called()

