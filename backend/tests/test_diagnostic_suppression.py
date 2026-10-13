"""
Test Diagnostic Access-Log & ESP32 Hardware Suppression
======================================================
Validates that:
1. Single-frame 1:1 endpoint (/iris/verify_1to1) with dry_run=True produces 0 AccessLog DB writes
   and 0 ESP32 serial commands, for both action="CHECK-IN" and action="CHECK-OUT".
2. Sequence 1:1 endpoint (/iris/verify_sequence_1to1) with dry_run=True produces 0 AccessLog DB writes
   and 0 ESP32 serial commands, for both action="CHECK-IN" and action="CHECK-OUT".
3. Direct service functions recognize_iris_v2_1to1_diagnostic and recognize_iris_v2_sequence_1to1_diagnostic
   honor explicit dry_run=True and produce 0 AccessLog writes regardless of action.
"""

import os
import sys
import io
from unittest.mock import patch
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
backend_root = os.path.join(ROOT, "backend")
if backend_root not in sys.path:
    sys.path.insert(0, backend_root)

from app.main import app
from app.database.connection import SessionLocal
from app.models.user_model import User, AccessLog
from app.services.iris_recognition_service import (
    recognize_iris_v2_1to1_diagnostic,
    recognize_iris_v2_sequence_1to1_diagnostic
)

client = TestClient(app)


def _make_jpeg_bytes(color=(40, 50, 60)):
    img = Image.new("RGB", (320, 240), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def get_log_count(db):
    return db.query(AccessLog).count()


def test_service_single_frame_dry_run_zero_db_writes():
    print("\n--- TEST: Service Single-Frame Dry-Run Zero DB Writes ---")
    db = SessionLocal()
    user = db.query(User).first()
    assert user is not None, "At least one user must exist in DB for test"
    user_id = user.id

    img_dummy = Image.new("RGB", (200, 200), color=(128, 128, 128))

    for action in ["CHECK-IN", "CHECK-OUT"]:
        initial_count = get_log_count(db)

        res = recognize_iris_v2_1to1_diagnostic(
            img_dummy,
            claimed_user_id=user_id,
            action=action,
            candidate_only=True,
            dry_run=True
        )

        final_count = get_log_count(db)
        assert final_count == initial_count, (
            f"AccessLog written during dry-run {action}! Initial: {initial_count}, Final: {final_count}"
        )
        assert res["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", (
            f"Expected SUPPRESSED_DIAGNOSTIC, got {res['hardware_action']}"
        )
        print(f"  [PASS] Service single-frame action={action} dry_run=True wrote 0 AccessLog entries.")
    db.close()


def test_service_sequence_dry_run_zero_db_writes():
    print("\n--- TEST: Service Sequence Dry-Run Zero DB Writes ---")
    db = SessionLocal()
    user = db.query(User).first()
    assert user is not None, "At least one user must exist in DB for test"
    user_id = user.id

    dummy_frames = [Image.new("RGB", (200, 200), color=(100, 100, 100)) for _ in range(3)]

    for action in ["CHECK-IN", "CHECK-OUT"]:
        initial_count = get_log_count(db)

        res = recognize_iris_v2_sequence_1to1_diagnostic(
            dummy_frames,
            claimed_user_id=user_id,
            action=action,
            dry_run=True
        )

        final_count = get_log_count(db)
        assert final_count == initial_count, (
            f"AccessLog written during sequence dry-run {action}! Initial: {initial_count}, Final: {final_count}"
        )
        assert res["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", (
            f"Expected SUPPRESSED_DIAGNOSTIC, got {res['hardware_action']}"
        )
        print(f"  [PASS] Service sequence action={action} dry_run=True wrote 0 AccessLog entries.")
    db.close()


def test_endpoint_single_frame_dry_run_zero_esp32_zero_db():
    print("\n--- TEST: Endpoint /iris/verify_1to1 Dry-Run Zero ESP32 & Zero DB ---")
    db = SessionLocal()
    user = db.query(User).first()
    assert user is not None, "At least one user must exist in DB for test"
    user_id = user.id

    jpeg_data = _make_jpeg_bytes()

    for action in ["CHECK-IN", "CHECK-OUT"]:
        initial_count = get_log_count(db)

        payload = {
            "claimed_user_id": user_id,
            "action": action,
            "dry_run": True
        }
        files = [("file", ("probe.jpg", jpeg_data, "image/jpeg"))]

        with patch("app.routers.iris_router.send_to_esp32") as mock_esp32:
            response = client.post("/iris/verify_1to1", data=payload, files=files)

        assert response.status_code == 200, response.text
        data = response.json()

        final_count = get_log_count(db)
        assert final_count == initial_count, (
            f"Endpoint /iris/verify_1to1 created AccessLog during dry_run=True action={action}!"
        )
        mock_esp32.assert_not_called()
        assert data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
        print(f"  [PASS] /iris/verify_1to1 action={action} dry_run=True -> 0 DB writes, 0 ESP32 calls.")
    db.close()


def test_endpoint_sequence_dry_run_zero_esp32_zero_db():
    print("\n--- TEST: Endpoint /iris/verify_sequence_1to1 Dry-Run Zero ESP32 & Zero DB ---")
    db = SessionLocal()
    user = db.query(User).first()
    assert user is not None, "At least one user must exist in DB for test"
    user_id = user.id

    jpeg_data = _make_jpeg_bytes()

    for action in ["CHECK-IN", "CHECK-OUT"]:
        initial_count = get_log_count(db)

        payload = {
            "claimed_user_id": user_id,
            "action": action,
            "dry_run": True
        }
        files = [
            ("files", ("f1.jpg", jpeg_data, "image/jpeg")),
            ("files", ("f2.jpg", jpeg_data, "image/jpeg")),
            ("files", ("f3.jpg", jpeg_data, "image/jpeg"))
        ]

        with patch("app.routers.iris_router.send_to_esp32") as mock_esp32:
            response = client.post("/iris/verify_sequence_1to1", data=payload, files=files)

        assert response.status_code == 200, response.text
        data = response.json()

        final_count = get_log_count(db)
        assert final_count == initial_count, (
            f"Endpoint /iris/verify_sequence_1to1 created AccessLog during dry_run=True action={action}!"
        )
        mock_esp32.assert_not_called()
        assert data.get("hardware_action") == "SUPPRESSED_DIAGNOSTIC"
        print(f"  [PASS] /iris/verify_sequence_1to1 action={action} dry_run=True -> 0 DB writes, 0 ESP32 calls.")
    db.close()


if __name__ == "__main__":
    print("================================================================")
    print(" RUNNING DIAGNOSTIC ACCESS-LOG & ESP32 SUPPRESSION TEST SUITE")
    print("================================================================")
    test_service_single_frame_dry_run_zero_db_writes()
    test_service_sequence_dry_run_zero_db_writes()
    test_endpoint_single_frame_dry_run_zero_esp32_zero_db()
    test_endpoint_sequence_dry_run_zero_esp32_zero_db()
    print("\n================================================================")
    print(" ALL DIAGNOSTIC SUPPRESSION TESTS PASSED SUCCESSFULLY!")
    print("================================================================")
