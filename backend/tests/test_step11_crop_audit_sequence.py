"""
Step 11 Verification Tests:
1. Crop validation & blank embedding rejection
2. Read-only audit service for existing templates
3. Coordinated sequence verification & dry_run hardware suppression
"""

import os
import sys
import numpy as np
from PIL import Image

# Setup sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "backend"))

from app.services.iris_detection_service import detect_and_crop_iris
from app.services.template_audit_service import audit_enrolled_templates
from app.services.face_recognition_service import (
    DEFAULT_FACE_MATCH_THRESHOLD,
    verify_two_stage_identity,
    verify_two_stage_sequence_identity
)


def test_1_invalid_crop_rejection():
    print("\n--- TEST 1: INVALID CROP REJECTION ---")
    blank_arr = np.zeros((300, 300, 3), dtype=np.uint8)
    res = detect_and_crop_iris(blank_arr)
    
    polar_ribbon = res[0]
    face_det = res[2]
    left_det = res[3]
    right_det = res[4]
    
    assert polar_ribbon is None, f"Expected polar_ribbon to be None on blank image, got {polar_ribbon}"
    assert face_det is False, "Expected face_det to be False"
    assert left_det is False, "Expected left_det to be False"
    assert right_det is False, "Expected right_det to be False"
    print(" [PASS] Blank image correctly rejected without fake center crop fallback.")


def test_2_read_only_template_audit():
    print("\n--- TEST 2: READ-ONLY TEMPLATE AUDIT ---")
    report = audit_enrolled_templates(verbose=False)
    
    assert report["status"] == "success", "Expected audit status to be success"
    assert "summary" in report, "Expected summary in report"
    assert report["summary"]["total_suspicious"] == 0, f"Expected 0 suspicious templates, got {report['summary']['total_suspicious']}"
    print(f" [PASS] Template audit completed read-only: {report['summary']['total_templates']} templates checked, {report['summary']['total_suspicious']} suspicious.")


def test_3_coordinated_sequence_dry_run():
    print("\n--- TEST 3: COORDINATED SEQUENCE & DRY_RUN SUPPRESSION ---")
    # Verify thresholds
    assert DEFAULT_FACE_MATCH_THRESHOLD == 0.360, f"Expected face threshold 0.360, got {DEFAULT_FACE_MATCH_THRESHOLD}"
    
    # Create 5 dummy frames to test sequence coordination and hardware suppression
    dummy_frames = [
        np.zeros((160, 160, 3), dtype=np.uint8) for _ in range(5)
    ]
    
    res = verify_two_stage_sequence_identity(
        dummy_frames,
        action="CHECK-IN",
        dry_run=True
    )
    
    assert res["final_decision"] == "REJECTED", "Expected dummy frames to be REJECTED"
    assert res["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", f"Expected SUPPRESSED_DIAGNOSTIC, got {res['hardware_action']}"
    assert res["face_stage"]["threshold"] == 0.360, f"Expected face threshold 0.360, got {res['face_stage']['threshold']}"
    assert res["iris_stage"]["threshold"] == 0.800, f"Expected iris threshold 0.800, got {res['iris_stage']['threshold']}"
    assert res["frames_evaluated"] == 5, f"Expected 5 frames evaluated, got {res['frames_evaluated']}"
    print(" [PASS] Coordinated sequence evaluated 5 frames, maintained thresholds (0.360/0.800), and suppressed hardware action.")


if __name__ == "__main__":
    print("========================================")
    print(" RUNNING STEP 11 FOCUSED VERIFICATION TESTS")
    print("========================================")
    test_1_invalid_crop_rejection()
    test_2_read_only_template_audit()
    test_3_coordinated_sequence_dry_run()
    print("\n========================================")
    print(" ALL STEP 11 TESTS PASSED SUCCESSFULLY!")
    print("========================================")
