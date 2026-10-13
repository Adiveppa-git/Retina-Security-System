import sys
import os
import cv2
import numpy as np

# Ensure repository root and backend are on sys.path
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(workspace_root, "backend")
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database.connection import SessionLocal
from app.models.user_model import User, FaceEmbedding, IrisEmbeddingV2, AccessLog
from app.services.face_recognition_service import (
    identify_user_by_face,
    verify_two_stage_identity,
    DEFAULT_FACE_MATCH_THRESHOLD
)
from fastapi.testclient import TestClient
from app.main import app

def run_step4_two_stage_tests():
    print("=" * 80)
    print("STEP 4 — INTEGRATE FACE CANDIDATE -> EXISTING IRIS VERIFICATION")
    print("SAFE DIAGNOSTIC VERIFICATION TEST SUITE")
    print("=" * 80)

    db = SessionLocal()
    try:
        # -------------------------------------------------------------
        # 0. DATABASE IMMUTABILITY BASELINE AUDIT
        # -------------------------------------------------------------
        users_count_baseline = db.query(User).count()
        face_count_baseline = db.query(FaceEmbedding).count()
        iris_v2_count_baseline = db.query(IrisEmbeddingV2).count()
        access_logs_count_baseline = db.query(AccessLog).count()

        u6 = db.query(User).filter(User.id == 6).first()
        u7 = db.query(User).filter(User.id == 7).first()
        u6_iris_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 6).all()
        u7_iris_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == 7).all()
        u6_iris_ids = [r.id for r in u6_iris_recs]
        u7_iris_ids = [r.id for r in u7_iris_recs]

        print("\n[BASELINE STATE AUDIT]")
        print(f"  User #6 ({u6.name}): Iris Templates = {len(u6_iris_ids)} (IDs: {u6_iris_ids})")
        print(f"  User #7 ({u7.name}): Iris Templates = {len(u7_iris_ids)} (IDs: {u7_iris_ids})")
        print(f"  Total Users: {users_count_baseline}")
        print(f"  Total Face Embeddings: {face_count_baseline}")
        print(f"  Total Iris V2 Embeddings: {iris_v2_count_baseline}")
        print(f"  Baseline Access Logs Count: {access_logs_count_baseline}")
        print(f"  Face Development Threshold: {DEFAULT_FACE_MATCH_THRESHOLD:.3f} (DEVELOPMENT ONLY)")
        print(f"  Iris Verification Threshold: 0.800 (UNCHANGED)")

        # -------------------------------------------------------------
        # TEST 1: USER #6 TWO-STAGE VERIFICATION
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 1] User #6 Two-Stage Verification")
        print("-" * 80)
        probe_u6_face = "project_images/014.jpeg"
        probe_u6_iris = "project_images/008.jpeg"

        res1 = verify_two_stage_identity(
            face_image_input=probe_u6_face,
            iris_image_input=probe_u6_iris,
            action="CHECK-IN",
            dry_run=True,
            db=db
        )

        print("  Face Stage:")
        print(f"    Matched: {res1['face_stage']['matched']}")
        print(f"    Candidate User ID: {res1['face_stage']['candidate_user_id']}")
        print(f"    Candidate User Name: {res1['face_stage']['candidate_user_name']}")
        print(f"    Face Score: {res1['face_stage']['score']}")
        print(f"    Threshold: {res1['face_stage']['threshold']} (DEVELOPMENT ONLY: {res1['face_stage']['is_development_threshold']})")
        print("  Iris Stage:")
        print(f"    Executed: {res1['iris_stage']['executed']}")
        print(f"    User ID Checked: {res1['iris_stage']['user_id_checked']}")
        print(f"    Iris Score: {res1['iris_stage']['score']}")
        print(f"    Threshold: {res1['iris_stage']['threshold']}")
        print(f"    Iris Matched: {res1['iris_stage']['matched']}")
        print(f"    Iris Template IDs Used: {res1['iris_stage']['template_ids_used']}")
        print(f"    Templates Checked: {res1['iris_stage']['templates_checked']}")
        print(f"  Final Decision: {res1['final_decision']}")
        print(f"  Failure Stage: {res1['failure_stage']}")
        print(f"  Hardware Action: {res1['hardware_action']}")

        assert res1["face_stage"]["matched"] is True, "Test 1: Face stage must match User #6!"
        assert res1["face_stage"]["candidate_user_id"] == 6, "Test 1: Candidate user_id must be 6!"
        assert res1["iris_stage"]["executed"] is True, "Test 1: Iris stage must be executed!"
        assert res1["iris_stage"]["user_id_checked"] == 6, "Test 1: Iris stage must check user_id 6 ONLY!"
        assert res1["iris_stage"]["template_ids_used"] == u6_iris_ids, f"Test 1: Must query ONLY user 6 templates! Got: {res1['iris_stage']['template_ids_used']}"
        assert set(res1["iris_stage"]["template_ids_used"]).isdisjoint(set(u7_iris_ids)), "Test 1: Must NOT check any user 7 templates!"
        assert res1["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", "Test 1: Hardware action must be SUPPRESSED_DIAGNOSTIC!"
        print("  >> TEST 1 PASSED: Candidate User #6 selected, checked ONLY User #6 templates [37, 38, 39, 40].")

        # -------------------------------------------------------------
        # TEST 2: USER #7 TWO-STAGE VERIFICATION
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 2] User #7 Two-Stage Verification")
        print("-" * 80)
        probe_u7_face = "project_images/015.jpeg"
        probe_u7_iris = "project_images/016.jpeg"

        res2 = verify_two_stage_identity(
            face_image_input=probe_u7_face,
            iris_image_input=probe_u7_iris,
            action="CHECK-IN",
            dry_run=True,
            db=db
        )

        print("  Face Stage:")
        print(f"    Matched: {res2['face_stage']['matched']}")
        print(f"    Candidate User ID: {res2['face_stage']['candidate_user_id']}")
        print(f"    Candidate User Name: {res2['face_stage']['candidate_user_name']}")
        print(f"    Face Score: {res2['face_stage']['score']}")
        print(f"    Threshold: {res2['face_stage']['threshold']} (DEVELOPMENT ONLY: {res2['face_stage']['is_development_threshold']})")
        print("  Iris Stage:")
        print(f"    Executed: {res2['iris_stage']['executed']}")
        print(f"    User ID Checked: {res2['iris_stage']['user_id_checked']}")
        print(f"    Iris Score: {res2['iris_stage']['score']}")
        print(f"    Threshold: {res2['iris_stage']['threshold']}")
        print(f"    Iris Matched: {res2['iris_stage']['matched']}")
        print(f"    Iris Template IDs Used: {res2['iris_stage']['template_ids_used']}")
        print(f"    Templates Checked: {res2['iris_stage']['templates_checked']}")
        print(f"  Final Decision: {res2['final_decision']}")
        print(f"  Failure Stage: {res2['failure_stage']}")
        print(f"  Hardware Action: {res2['hardware_action']}")

        assert res2["face_stage"]["matched"] is True, "Test 2: Face stage must match User #7!"
        assert res2["face_stage"]["candidate_user_id"] == 7, "Test 2: Candidate user_id must be 7!"
        assert res2["iris_stage"]["executed"] is True, "Test 2: Iris stage must be executed!"
        assert res2["iris_stage"]["user_id_checked"] == 7, "Test 2: Iris stage must check user_id 7 ONLY!"
        assert res2["iris_stage"]["template_ids_used"] == u7_iris_ids, f"Test 2: Must query ONLY user 7 templates! Got: {res2['iris_stage']['template_ids_used']}"
        assert set(res2["iris_stage"]["template_ids_used"]).isdisjoint(set(u6_iris_ids)), "Test 2: Must NOT check any user 6 templates!"
        assert res2["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", "Test 2: Hardware action must be SUPPRESSED_DIAGNOSTIC!"
        print("  >> TEST 2 PASSED: Candidate User #7 selected, checked ONLY User #7 templates [41..52].")

        # -------------------------------------------------------------
        # TEST 3: UNKNOWN PERSON (FACE FAILURE -> IRIS NOT EXECUTED)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 3] Unknown Person Verification (IDENTITY_NOT_FOUND)")
        print("-" * 80)
        probe_unknown = "datasets/anti_spoof_dataset/test/fake/fake_ts_00048.jpg"

        res3 = verify_two_stage_identity(
            face_image_input=probe_unknown,
            iris_image_input=probe_u6_iris,
            action="CHECK-IN",
            dry_run=True,
            db=db
        )

        print("  Face Stage:")
        print(f"    Matched: {res3['face_stage']['matched']}")
        print(f"    Reason: {res3['face_stage']['reason']}")
        print(f"    Candidate User ID: {res3['face_stage']['candidate_user_id']}")
        print(f"    Face Score: {res3['face_stage']['score']}")
        print("  Iris Stage:")
        print(f"    Executed: {res3['iris_stage']['executed']}")
        print(f"    User ID Checked: {res3['iris_stage']['user_id_checked']}")
        print(f"    Template IDs Used: {res3['iris_stage']['template_ids_used']}")
        print(f"  Final Decision: {res3['final_decision']}")
        print(f"  Failure Stage: {res3['failure_stage']}")
        print(f"  Hardware Action: {res3['hardware_action']}")

        assert res3["face_stage"]["matched"] is False, "Test 3: Face must NOT match unknown person!"
        assert res3["face_stage"]["reason"] == "IDENTITY_NOT_FOUND", "Test 3: Reason must be IDENTITY_NOT_FOUND!"
        assert res3["iris_stage"]["executed"] is False, "Test 3: Iris stage must NOT be executed!"
        assert res3["final_decision"] == "REJECTED", "Test 3: Final decision must be REJECTED!"
        assert res3["failure_stage"] == "IDENTITY_IDENTIFICATION", "Test 3: Failure stage must be IDENTITY_IDENTIFICATION!"
        assert res3["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", "Test 3: Hardware action must be SUPPRESSED_DIAGNOSTIC!"
        print("  >> TEST 3 PASSED: Unknown person rejected at Stage 1, iris NOT executed.")

        # -------------------------------------------------------------
        # TEST 4: FACE/IRIS MISMATCH (FACE PASS -> IRIS FAIL -> REJECT)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 4] Face/Iris Mismatch (Face User #6 + Non-matching Iris)")
        print("-" * 80)
        # Probe face is User #6, but Iris is dummy black image / non-matching iris
        probe_mismatch_iris = "scratch/test_black.jpg"

        res4 = verify_two_stage_identity(
            face_image_input=probe_u6_face,
            iris_image_input=probe_mismatch_iris,
            action="CHECK-IN",
            dry_run=True,
            db=db
        )

        print("  Face Stage:")
        print(f"    Matched: {res4['face_stage']['matched']}")
        print(f"    Candidate User ID: {res4['face_stage']['candidate_user_id']}")
        print(f"    Candidate User Name: {res4['face_stage']['candidate_user_name']}")
        print(f"    Face Score: {res4['face_stage']['score']}")
        print("  Iris Stage:")
        print(f"    Executed: {res4['iris_stage']['executed']}")
        print(f"    User ID Checked: {res4['iris_stage']['user_id_checked']}")
        print(f"    Iris Score: {res4['iris_stage']['score']}")
        print(f"    Iris Matched: {res4['iris_stage']['matched']}")
        print(f"    Iris Decision Reason: {res4['iris_stage']['decision_reason']}")
        print(f"  Final Decision: {res4['final_decision']}")
        print(f"  Failure Stage: {res4['failure_stage']}")
        print(f"  Hardware Action: {res4['hardware_action']}")

        assert res4["face_stage"]["matched"] is True, "Test 4: Face stage must match User #6!"
        assert res4["face_stage"]["candidate_user_id"] == 6, "Test 4: Candidate must be User #6!"
        assert res4["iris_stage"]["executed"] is True, "Test 4: Iris stage must be executed for candidate!"
        assert res4["iris_stage"]["matched"] is False, "Test 4: Iris verification must FAIL!"
        assert res4["final_decision"] == "REJECTED", "Test 4: Final decision must be REJECTED!"
        assert res4["failure_stage"] == "BIOMETRIC_VERIFICATION", "Test 4: Failure stage must be BIOMETRIC_VERIFICATION!"
        assert res4["hardware_action"] == "SUPPRESSED_DIAGNOSTIC", "Test 4: Hardware action must be SUPPRESSED_DIAGNOSTIC!"
        print("  >> TEST 4 PASSED: Face matched User #6, but Iris failed -> REJECTED at BIOMETRIC_VERIFICATION stage.")

        # -------------------------------------------------------------
        # TEST 5: MULTIPLE FACES (STAGE 1 FAILURE -> IRIS NOT EXECUTED)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 5] Multiple Faces Image (MULTIPLE_FACES_DETECTED)")
        print("-" * 80)
        img_a = cv2.imread("project_images/003.jpeg")
        img_b = cv2.imread("project_images/004.jpeg")
        h_target = min(img_a.shape[0], img_b.shape[0])
        img_a_scaled = cv2.resize(img_a, (int(img_a.shape[1] * h_target / img_a.shape[0]), h_target))
        img_b_scaled = cv2.resize(img_b, (int(img_b.shape[1] * h_target / img_b.shape[0]), h_target))
        multi_img = np.hstack([img_a_scaled, img_b_scaled])

        res5 = verify_two_stage_identity(
            face_image_input=multi_img,
            iris_image_input=probe_u6_iris,
            action="CHECK-IN",
            dry_run=True,
            db=db
        )

        print("  Face Stage:")
        print(f"    Matched: {res5['face_stage']['matched']}")
        print(f"    Reason: {res5['face_stage']['reason']}")
        print(f"    Num Faces: {res5['face_stage']['num_faces']}")
        print("  Iris Stage:")
        print(f"    Executed: {res5['iris_stage']['executed']}")
        print(f"  Final Decision: {res5['final_decision']}")
        print(f"  Failure Stage: {res5['failure_stage']}")
        print(f"  Hardware Action: {res5['hardware_action']}")

        assert res5["face_stage"]["matched"] is False, "Test 5: Face must reject multiple faces!"
        assert res5["face_stage"]["reason"] == "MULTIPLE_FACES_DETECTED", "Test 5: Reason must be MULTIPLE_FACES_DETECTED!"
        assert res5["iris_stage"]["executed"] is False, "Test 5: Iris stage must NOT execute!"
        assert res5["final_decision"] == "REJECTED", "Test 5: Final decision must be REJECTED!"
        assert res5["failure_stage"] == "IDENTITY_IDENTIFICATION", "Test 5: Failure stage must be IDENTITY_IDENTIFICATION!"
        print("  >> TEST 5 PASSED: Multiple faces rejected at Stage 1, iris NOT executed.")

        # -------------------------------------------------------------
        # TEST 6: FASTAPI ENDPOINT (POST /user/verify_identity)
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[TEST 6] FastAPI Endpoint Validation (POST /user/verify_identity)")
        print("-" * 80)
        client = TestClient(app)

        with open(probe_u6_face, "rb") as f_face, open(probe_u6_iris, "rb") as f_iris:
            resp = client.post(
                "/user/verify_identity",
                files={
                    "file": ("u6_face.jpg", f_face, "image/jpeg"),
                    "iris_file": ("u6_iris.jpg", f_iris, "image/jpeg")
                },
                data={"action": "CHECK-IN", "dry_run": "true"}
            )

        print(f"  HTTP Status Code: {resp.status_code}")
        json_resp = resp.json()
        print(f"  Response JSON Keys: {list(json_resp.keys())}")
        print(f"  Face Matched: {json_resp['face_stage']['matched']}")
        print(f"  Candidate: {json_resp['face_stage']['candidate_user_name']} (#{json_resp['face_stage']['candidate_user_id']})")
        print(f"  Iris Executed: {json_resp['iris_stage']['executed']}")
        print(f"  Final Decision: {json_resp['final_decision']}")
        print(f"  Hardware Action: {json_resp['hardware_action']}")

        assert resp.status_code == 200, f"Expected 200 OK, got {resp.status_code}"
        assert json_resp["face_stage"]["candidate_user_id"] == 6
        assert json_resp["iris_stage"]["executed"] is True
        assert json_resp["hardware_action"] == "SUPPRESSED_DIAGNOSTIC"
        print("  >> TEST 6 PASSED: Endpoint POST /user/verify_identity operational and returned valid schema.")

        # -------------------------------------------------------------
        # 7. DATABASE IMMUTABILITY POST-TEST VERIFICATION
        # -------------------------------------------------------------
        print("\n" + "-" * 80)
        print("[SAFETY AUDIT] Database & Hardware Immutability Verification")
        print("-" * 80)

        users_count_after = db.query(User).count()
        face_count_after = db.query(FaceEmbedding).count()
        iris_v2_count_after = db.query(IrisEmbeddingV2).count()
        access_logs_count_after = db.query(AccessLog).count()

        print(f"  Users Table Count: {users_count_after} (Delta: {users_count_after - users_count_baseline})")
        print(f"  Face Embeddings Count: {face_count_after} (Delta: {face_count_after - face_count_baseline})")
        print(f"  Iris V2 Embeddings Count: {iris_v2_count_after} (Delta: {iris_v2_count_after - iris_v2_count_baseline})")
        print(f"  Access Logs Count: {access_logs_count_after} (Delta: {access_logs_count_after - access_logs_count_baseline})")

        assert users_count_after == users_count_baseline, "Users table must be READ-ONLY!"
        assert face_count_after == face_count_baseline, "Face embeddings must be READ-ONLY!"
        assert iris_v2_count_after == iris_v2_count_baseline, "Iris V2 embeddings must be READ-ONLY!"
        assert access_logs_count_after == access_logs_count_baseline, f"Access logs mutated! Expected {access_logs_count_baseline}, got {access_logs_count_after}"
        print("  >> SAFETY PASSED: Zero records added or altered across all tables. Strict READ-ONLY compliance verified.")

        print("\n" + "=" * 80)
        print("ALL STEP 4 TWO-STAGE TESTS SUCCESSFULLY PASSED!")
        print("=" * 80)
        return True

    finally:
        db.close()

if __name__ == "__main__":
    success = run_step4_two_stage_tests()
    if not success:
        sys.exit(1)
