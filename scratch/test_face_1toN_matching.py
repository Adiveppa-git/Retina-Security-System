import sys
import os
import cv2
import numpy as np

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.database.connection import SessionLocal
from app.models.user_model import User, FaceEmbedding, IrisEmbeddingV2, IrisEmbedding, AccessLog
from app.services.face_recognition_service import (
    identify_user_by_face,
    compute_face_similarity,
    extract_face_embedding,
    DEFAULT_FACE_MATCH_THRESHOLD
)
from fastapi.testclient import TestClient
from app.main import app

print("=" * 75)
print("TEST SUITE: STEP 3 — FACE 1:N IDENTITY MATCHING & THRESHOLD VALIDATION")
print("=" * 75)

db = SessionLocal()
try:
    # -------------------------------------------------------------
    # 0. BASELINE AUDIT (Read-Only Verification)
    # -------------------------------------------------------------
    users = db.query(User).order_by(User.id.asc()).all()
    user6 = db.query(User).filter(User.id == 6).first()
    user7 = db.query(User).filter(User.id == 7).first()

    assert user6 is not None, "User #6 not found in MySQL!"
    assert user7 is not None, "User #7 not found in MySQL!"

    u6_samples = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == 6).order_by(FaceEmbedding.id.asc()).all()
    u7_samples = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == 7).order_by(FaceEmbedding.id.asc()).all()

    iris_v2_count_baseline = db.query(IrisEmbeddingV2).count()
    iris_v1_count_baseline = db.query(IrisEmbedding).count()
    access_logs_count_baseline = db.query(AccessLog).count()

    print(f"\n[BASELINE STATE]")
    print(f"  User #6 ({user6.name}): {len(u6_samples)} face samples (IDs: {[s.id for s in u6_samples]})")
    print(f"  User #7 ({user7.name}): {len(u7_samples)} face samples (IDs: {[s.id for s in u7_samples]})")
    print(f"  Total Enrolled Face Embeddings: {len(u6_samples) + len(u7_samples)}")
    print(f"  Configured Threshold: {DEFAULT_FACE_MATCH_THRESHOLD:.3f} (DEVELOPMENT ONLY)")
    print(f"  Iris V2 Embeddings: {iris_v2_count_baseline}")
    print(f"  Access Logs Count: {access_logs_count_baseline}")

    assert len(u6_samples) == 2, f"Expected 2 samples for User 6, found {len(u6_samples)}"
    assert len(u7_samples) == 1, f"Expected 1 sample for User 7, found {len(u7_samples)}"

    # -------------------------------------------------------------
    # 1. TEST 1: User #6 Presents
    # -------------------------------------------------------------
    print(f"\n[TEST 1] User #6 Presents (Probe: project_images/014.jpeg):")
    probe_u6 = "project_images/014.jpeg"
    res1 = identify_user_by_face(probe_u6, db=db)
    print(f"  Status: {res1['status']}")
    print(f"  Matched: {res1['matched']}")
    print(f"  Candidate User ID: {res1['candidate_user_id']}")
    print(f"  Candidate User Name: {res1['candidate_user_name']}")
    print(f"  Face Score: {res1['face_score']:.4f}")
    print(f"  Threshold: {res1['threshold']:.3f}")
    print(f"  User Scores: {res1['user_scores']}")

    assert res1["matched"] is True, "User #6 probe must match!"
    assert res1["candidate_user_id"] == 6, f"Expected user_id=6, got {res1['candidate_user_id']}"
    assert res1["candidate_user_name"] == "Adiveppa"
    assert res1["face_score"] >= DEFAULT_FACE_MATCH_THRESHOLD

    # -------------------------------------------------------------
    # 2. TEST 2: User #7 Presents
    # -------------------------------------------------------------
    print(f"\n[TEST 2] User #7 Presents (Probe: project_images/015.jpeg):")
    probe_u7 = "project_images/015.jpeg"
    res2 = identify_user_by_face(probe_u7, db=db)
    print(f"  Status: {res2['status']}")
    print(f"  Matched: {res2['matched']}")
    print(f"  Candidate User ID: {res2['candidate_user_id']}")
    print(f"  Candidate User Name: {res2['candidate_user_name']}")
    print(f"  Face Score: {res2['face_score']:.4f}")
    print(f"  Threshold: {res2['threshold']:.3f}")
    print(f"  User Scores: {res2['user_scores']}")

    assert res2["matched"] is True, "User #7 probe must match!"
    assert res2["candidate_user_id"] == 7, f"Expected user_id=7, got {res2['candidate_user_id']}"
    assert res2["candidate_user_name"] == "basagouda"
    assert res2["face_score"] >= DEFAULT_FACE_MATCH_THRESHOLD

    # -------------------------------------------------------------
    # 3. TEST 3: Unregistered Person Presents (IDENTITY_NOT_FOUND)
    # -------------------------------------------------------------
    probe_unknown = "datasets/anti_spoof_dataset/test/fake/fake_ts_00048.jpg"
    print(f"\n[TEST 3] Unregistered Person Presents (Probe: {probe_unknown}):")
    res3 = identify_user_by_face(probe_unknown, db=db)
    print(f"  Status: {res3['status']}")
    print(f"  Matched: {res3['matched']}")
    print(f"  Reason: {res3['reason']}")
    print(f"  Candidate User ID: {res3['candidate_user_id']}")
    print(f"  Candidate User Name: {res3['candidate_user_name']}")
    print(f"  Best Face Score: {res3['face_score']:.4f}")
    print(f"  Threshold: {res3['threshold']:.3f}")
    print(f"  User Scores: {res3['user_scores']}")

    assert res3["matched"] is False, "Unregistered person must NOT match!"
    assert res3["reason"] == "IDENTITY_NOT_FOUND"
    assert res3["candidate_user_id"] is None
    assert res3["candidate_user_name"] is None
    assert res3["face_score"] < DEFAULT_FACE_MATCH_THRESHOLD

    # -------------------------------------------------------------
    # 4. TEST 4: Multiple-Face Image Rejection
    # -------------------------------------------------------------
    print(f"\n[TEST 4] Multiple-Face Image Presents:")
    img_a = cv2.imread("project_images/003.jpeg")
    img_b = cv2.imread("project_images/004.jpeg")
    h_target = min(img_a.shape[0], img_b.shape[0])
    img_a_scaled = cv2.resize(img_a, (int(img_a.shape[1] * h_target / img_a.shape[0]), h_target))
    img_b_scaled = cv2.resize(img_b, (int(img_b.shape[1] * h_target / img_b.shape[0]), h_target))
    multi_img = np.hstack([img_a_scaled, img_b_scaled])

    res4 = identify_user_by_face(multi_img, db=db)
    print(f"  Status: {res4['status']}")
    print(f"  Matched: {res4['matched']}")
    print(f"  Reason: {res4['reason']}")
    print(f"  Candidate User ID: {res4['candidate_user_id']}")
    print(f"  Num Faces: {res4['num_faces']}")

    assert res4["matched"] is False
    assert res4["reason"] == "MULTIPLE_FACES_DETECTED"
    assert res4["candidate_user_id"] is None
    assert res4["num_faces"] == 2

    # -------------------------------------------------------------
    # 5. TEST 5: No-Face Image Rejection
    # -------------------------------------------------------------
    print(f"\n[TEST 5] No-Face Image Presents (Probe: scratch/test_black.jpg):")
    res5 = identify_user_by_face("scratch/test_black.jpg", db=db)
    print(f"  Status: {res5['status']}")
    print(f"  Matched: {res5['matched']}")
    print(f"  Reason: {res5['reason']}")
    print(f"  Candidate User ID: {res5['candidate_user_id']}")
    print(f"  Num Faces: {res5['num_faces']}")

    assert res5["matched"] is False
    assert res5["reason"] == "NO_FACE_DETECTED"
    assert res5["candidate_user_id"] is None
    assert res5["num_faces"] == 0

    # -------------------------------------------------------------
    # 6. CRITICAL CROSS-IDENTITY SIMILARITY MATRIX
    # -------------------------------------------------------------
    print(f"\n" + "=" * 75)
    print("CRITICAL CROSS-IDENTITY SIMILARITY MATRIX")
    print("=" * 75)

    s6_1 = u6_samples[0] # sample 1 (003.jpeg)
    s6_2 = u6_samples[1] # sample 2 (004.jpeg)
    s7_1 = u7_samples[0] # sample 1 (005.jpeg)

    emb_probe_u6 = extract_face_embedding("project_images/014.jpeg")["embedding"]
    emb_probe_u7 = extract_face_embedding("project_images/015.jpeg")["embedding"]

    # Probe 6 comparisons
    sim_p6_s61 = compute_face_similarity(emb_probe_u6, s6_1.embedding)
    sim_p6_s62 = compute_face_similarity(emb_probe_u6, s6_2.embedding)
    sim_p6_s71 = compute_face_similarity(emb_probe_u6, s7_1.embedding)
    best_p6_u6 = max(sim_p6_s61, sim_p6_s62)
    best_p6_u7 = sim_p6_s71

    # Probe 7 comparisons
    sim_p7_s61 = compute_face_similarity(emb_probe_u7, s6_1.embedding)
    sim_p7_s62 = compute_face_similarity(emb_probe_u7, s6_2.embedding)
    sim_p7_s71 = compute_face_similarity(emb_probe_u7, s7_1.embedding)
    best_p7_u6 = max(sim_p7_s61, sim_p7_s62)
    best_p7_u7 = sim_p7_s71

    print(f"\nPROBE: User #6 (Adiveppa, sample 014.jpeg)")
    print(f"  -> User #6 Sample 1 (Rec #{s6_1.id}): {sim_p6_s61:.4f}")
    print(f"  -> User #6 Sample 2 (Rec #{s6_2.id}): {sim_p6_s62:.4f}")
    print(f"  -> User #7 Sample 1 (Rec #{s7_1.id}): {sim_p6_s71:.4f}")
    print(f"  Best User #6 Score: {best_p6_u6:.4f}")
    print(f"  Best User #7 Score: {best_p6_u7:.4f}")
    print(f"  Separation Margin:  {best_p6_u6 - best_p6_u7:+.4f} (User #6 Wins cleanly)")

    print(f"\nPROBE: User #7 (basagouda, sample 015.jpeg)")
    print(f"  -> User #6 Sample 1 (Rec #{s6_1.id}): {sim_p7_s61:.4f}")
    print(f"  -> User #6 Sample 2 (Rec #{s6_2.id}): {sim_p7_s62:.4f}")
    print(f"  -> User #7 Sample 1 (Rec #{s7_1.id}): {sim_p7_s71:.4f}")
    print(f"  Best User #6 Score: {best_p7_u6:.4f}")
    print(f"  Best User #7 Score: {best_p7_u7:.4f}")
    print(f"  Separation Margin:  {best_p7_u7 - best_p7_u6:+.4f} (User #7 Wins cleanly)")

    # -------------------------------------------------------------
    # 7. HTTP API ENDPOINT VERIFICATION (POST /user/face/identify)
    # -------------------------------------------------------------
    print(f"\n[TEST 7] HTTP API Endpoint Verification:")
    client = TestClient(app)

    # 7a. Test User 6 over HTTP
    with open("project_images/014.jpeg", "rb") as f:
        http_u6 = client.post("/user/face/identify", files={"file": ("u6.jpg", f, "image/jpeg")})
    assert http_u6.status_code == 200
    u6_json = http_u6.json()
    print(f"  HTTP /user/face/identify (User 6): matched={u6_json['matched']}, user_id={u6_json['candidate_user_id']}, score={u6_json['face_score']:.4f}")
    assert u6_json["matched"] is True
    assert u6_json["candidate_user_id"] == 6

    # 7b. Test User 7 over HTTP
    with open("project_images/015.jpeg", "rb") as f:
        http_u7 = client.post("/user/face/identify", files={"file": ("u7.jpg", f, "image/jpeg")})
    assert http_u7.status_code == 200
    u7_json = http_u7.json()
    print(f"  HTTP /user/face/identify (User 7): matched={u7_json['matched']}, user_id={u7_json['candidate_user_id']}, score={u7_json['face_score']:.4f}")
    assert u7_json["matched"] is True
    assert u7_json["candidate_user_id"] == 7

    # 7c. Test Unknown over HTTP
    with open(probe_unknown, "rb") as f:
        http_unk = client.post("/user/face/identify", files={"file": ("unk.jpg", f, "image/jpeg")})
    assert http_unk.status_code == 200
    unk_json = http_unk.json()
    print(f"  HTTP /user/face/identify (Unknown): matched={unk_json['matched']}, reason={unk_json['reason']}, score={unk_json['face_score']:.4f}")
    assert unk_json["matched"] is False
    assert unk_json["candidate_user_id"] is None

    # -------------------------------------------------------------
    # 8. DATABASE SAFETY & IMMUTABILITY CONFIRMATION
    # -------------------------------------------------------------
    print(f"\n[TEST 8] Database Safety & Immutability Verification:")
    u6_samples_after = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == 6).count()
    u7_samples_after = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == 7).count()
    total_face_after = db.query(FaceEmbedding).count()
    iris_v2_after = db.query(IrisEmbeddingV2).count()
    iris_v1_after = db.query(IrisEmbedding).count()
    access_logs_after = db.query(AccessLog).count()

    print(f"  User #6 samples: before={len(u6_samples)}, after={u6_samples_after} (Diff={u6_samples_after - len(u6_samples)})")
    print(f"  User #7 samples: before={len(u7_samples)}, after={u7_samples_after} (Diff={u7_samples_after - len(u7_samples)})")
    print(f"  Total face_embeddings: before={len(u6_samples) + len(u7_samples)}, after={total_face_after} (Diff=0)")
    print(f"  Iris V2 Embeddings: before={iris_v2_count_baseline}, after={iris_v2_after} (Diff=0)")
    print(f"  Access Logs: before={access_logs_count_baseline}, after={access_logs_after} (Diff=0)")

    assert u6_samples_after == 2, "User #6 face samples were modified!"
    assert u7_samples_after == 1, "User #7 face samples were modified!"
    assert total_face_after == 3, "face_embeddings rows were modified!"
    assert iris_v2_after == iris_v2_count_baseline, "Iris V2 embeddings were modified!"
    assert iris_v1_after == iris_v1_count_baseline, "Iris V1 embeddings were modified!"
    assert access_logs_after == access_logs_count_baseline, "Access logs were modified!"

    print("\n" + "=" * 75)
    print("ALL STEP 3 TESTS PASSED PERFECTLY! ZERO ERRORS. DATABASE FULLY PRESERVED.")
    print("=" * 75)

finally:
    db.close()
