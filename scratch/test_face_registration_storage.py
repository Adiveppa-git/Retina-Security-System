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
    register_face_sample,
    get_user_face_samples,
    extract_face_embedding
)
from fastapi.testclient import TestClient
from app.main import app

print("=" * 70)
print("TEST SUITE: STEP 2 — FACE REGISTRATION STORAGE & ENDPOINTS")
print("=" * 70)

db = SessionLocal()
try:
    # -------------------------------------------------------------
    # 0. INITIALIZE CLEAN STATE FOR face_embeddings ONLY
    # -------------------------------------------------------------
    db.query(FaceEmbedding).delete()
    db.commit()

    users = db.query(User).order_by(User.id.asc()).all()
    assert len(users) >= 2, f"Expected at least 2 users, found {len(users)}"
    user1 = users[0]
    user2 = users[1]

    iris_v2_count_before = db.query(IrisEmbeddingV2).count()
    iris_v1_count_before = db.query(IrisEmbedding).count()
    access_logs_count_before = db.query(AccessLog).count()
    face_count_before = db.query(FaceEmbedding).count()

    print(f"\n[BASELINE STATE]")
    print(f"  User #1: ID #{user1.id} ({user1.name})")
    print(f"  User #2: ID #{user2.id} ({user2.name})")
    print(f"  Iris V2 Embeddings count: {iris_v2_count_before}")
    print(f"  Iris V1 Embeddings count: {iris_v1_count_before}")
    print(f"  Access Logs count: {access_logs_count_before}")
    print(f"  Face Embeddings count: {face_count_before}")

    # -------------------------------------------------------------
    # 1. Register First Face Sample for User #1
    # -------------------------------------------------------------
    print(f"\n[TEST 1] Register First Face Sample for User #1 (ID #{user1.id}):")
    with open("project_images/003.jpeg", "rb") as f:
        img_bytes1 = f.read()

    res1 = register_face_sample(user_id=user1.id, image_data=img_bytes1, db=db)
    print(f"  Status: {res1['status']}")
    print(f"  Sample ID: {res1.get('sample_id')}")
    print(f"  Total Samples for User #1: {res1.get('total_face_samples')}")
    print(f"  Confidence: {res1.get('confidence'):.4f}")
    print(f"  Embedding Dim: {res1.get('embedding_dim')}")
    print(f"  L2 Norm: {res1.get('l2_norm'):.6f}")

    assert res1["status"] == "success", f"Expected success, got {res1}"
    assert res1["user_id"] == user1.id
    assert res1["embedding_dim"] == 128
    assert abs(res1["l2_norm"] - 1.0) < 1e-3
    assert res1["total_face_samples"] == 1

    # -------------------------------------------------------------
    # 2. Register Second Face Sample for User #1 (Multiple Samples Test)
    # -------------------------------------------------------------
    print(f"\n[TEST 2] Register Second Face Sample for User #1 (ID #{user1.id}):")
    with open("project_images/004.jpeg", "rb") as f:
        img_bytes2 = f.read()

    res2 = register_face_sample(user_id=user1.id, image_data=img_bytes2, db=db)
    print(f"  Status: {res2['status']}")
    print(f"  Sample ID: {res2.get('sample_id')}")
    print(f"  Total Samples for User #1: {res2.get('total_face_samples')}")
    print(f"  Confidence: {res2.get('confidence'):.4f}")
    print(f"  Embedding Dim: {res2.get('embedding_dim')}")
    print(f"  L2 Norm: {res2.get('l2_norm'):.6f}")

    assert res2["status"] == "success", f"Expected success, got {res2}"
    assert res2["user_id"] == user1.id
    assert res2["sample_id"] != res1["sample_id"], "Each sample must have a unique record ID!"
    assert res2["total_face_samples"] == 2, "User #1 should now have exactly 2 samples"

    # -------------------------------------------------------------
    # 3. Register One Face Sample for User #2 (ID #{user2.id})
    # -------------------------------------------------------------
    print(f"\n[TEST 3] Register One Face Sample for User #2 (ID #{user2.id}):")
    with open("project_images/005.jpeg", "rb") as f:
        img_bytes3 = f.read()

    res3 = register_face_sample(user_id=user2.id, image_data=img_bytes3, db=db)
    print(f"  Status: {res3['status']}")
    print(f"  Sample ID: {res3.get('sample_id')}")
    print(f"  Total Samples for User #2: {res3.get('total_face_samples')}")
    print(f"  Confidence: {res3.get('confidence'):.4f}")
    print(f"  Embedding Dim: {res3.get('embedding_dim')}")
    print(f"  L2 Norm: {res3.get('l2_norm'):.6f}")

    assert res3["status"] == "success", f"Expected success, got {res3}"
    assert res3["user_id"] == user2.id
    assert res3["total_face_samples"] == 1

    # -------------------------------------------------------------
    # 4. Invalid / No-Face Image Rejection Test
    # -------------------------------------------------------------
    print(f"\n[TEST 4] Invalid / No-Face Image Rejection:")
    with open("scratch/test_black.jpg", "rb") as f:
        no_face_bytes = f.read()

    rows_before_rejection = db.query(FaceEmbedding).count()
    res_no_face = register_face_sample(user_id=user1.id, image_data=no_face_bytes, db=db)
    rows_after_rejection = db.query(FaceEmbedding).count()

    print(f"  Status: {res_no_face['status']}")
    print(f"  Error Code: {res_no_face.get('error_code')}")
    print(f"  Message: {res_no_face.get('message')}")
    print(f"  DB Rows Before: {rows_before_rejection} | DB Rows After: {rows_after_rejection}")

    assert res_no_face["status"] == "error"
    assert res_no_face["error_code"] == "NO_FACE_DETECTED"
    assert rows_before_rejection == rows_after_rejection, "No row should be inserted on detection failure!"

    # -------------------------------------------------------------
    # 5. Multiple-Face Image Rejection Test
    # -------------------------------------------------------------
    print(f"\n[TEST 5] Multiple-Face Image Rejection:")
    img_a = cv2.imread("project_images/003.jpeg")
    img_b = cv2.imread("project_images/004.jpeg")
    h_target = min(img_a.shape[0], img_b.shape[0])
    img_a_scaled = cv2.resize(img_a, (int(img_a.shape[1] * h_target / img_a.shape[0]), h_target))
    img_b_scaled = cv2.resize(img_b, (int(img_b.shape[1] * h_target / img_b.shape[0]), h_target))
    multi_img = np.hstack([img_a_scaled, img_b_scaled])
    _, multi_buf = cv2.imencode(".jpg", multi_img)
    multi_bytes = multi_buf.tobytes()

    rows_before_multi = db.query(FaceEmbedding).count()
    res_multi = register_face_sample(user_id=user1.id, image_data=multi_bytes, db=db)
    rows_after_multi = db.query(FaceEmbedding).count()

    print(f"  Status: {res_multi['status']}")
    print(f"  Error Code: {res_multi.get('error_code')}")
    print(f"  Message: {res_multi.get('message')}")
    print(f"  DB Rows Before: {rows_before_multi} | DB Rows After: {rows_after_multi}")

    assert res_multi["status"] == "error"
    assert res_multi["error_code"] == "MULTIPLE_FACES_DETECTED"
    assert rows_before_multi == rows_after_multi, "No row should be inserted for multiple faces!"

    # -------------------------------------------------------------
    # 6. Non-Existent User ID Rejection Test
    # -------------------------------------------------------------
    print(f"\n[TEST 6] Non-Existent User ID Rejection:")
    res_bad_user = register_face_sample(user_id=999999, image_data=img_bytes1, db=db)
    print(f"  Status: {res_bad_user['status']}")
    print(f"  Error Code: {res_bad_user.get('error_code')}")
    assert res_bad_user["status"] == "error"
    assert res_bad_user["error_code"] == "USER_NOT_FOUND"

    # -------------------------------------------------------------
    # 7. HTTP API Endpoint Verification via TestClient
    # -------------------------------------------------------------
    print(f"\n[TEST 7] HTTP API Endpoints Test (TestClient):")
    client = TestClient(app)

    # Test GET samples for user 1
    get1 = client.get(f"/user/{user1.id}/face/samples")
    assert get1.status_code == 200
    get1_json = get1.json()
    print(f"  GET /user/{user1.id}/face/samples: total={get1_json['total_samples']}")
    assert get1_json["total_samples"] == 2

    # Test GET samples for user 2
    get2 = client.get(f"/user/{user2.id}/face/samples")
    assert get2.status_code == 200
    get2_json = get2.json()
    print(f"  GET /user/{user2.id}/face/samples: total={get2_json['total_samples']}")
    assert get2_json["total_samples"] == 1

    # Test invalid upload rejected over HTTP
    bad_upload = client.post(f"/user/{user1.id}/face/register")
    assert bad_upload.status_code in [200, 422]  # handled error

    # -------------------------------------------------------------
    # 8. Database Records Verification (Dimensions, Normalization, User IDs)
    # -------------------------------------------------------------
    print(f"\n[TEST 8] Database Records Deep Verification:")
    db.commit()  # refresh session snapshot
    all_face_records = db.query(FaceEmbedding).order_by(FaceEmbedding.id.asc()).all()
    print(f"  Total face_embeddings rows: {len(all_face_records)}")
    assert len(all_face_records) == 3, f"Expected 3 records, got {len(all_face_records)}"

    u1_samples = [r for r in all_face_records if r.user_id == user1.id]
    u2_samples = [r for r in all_face_records if r.user_id == user2.id]
    print(f"  User #{user1.id} ({user1.name}) samples: {len(u1_samples)}")
    print(f"  User #{user2.id} ({user2.name}) samples: {len(u2_samples)}")

    assert len(u1_samples) == 2, f"Expected 2 samples for User #1, got {len(u1_samples)}"
    assert len(u2_samples) == 1, f"Expected 1 sample for User #2, got {len(u2_samples)}"

    for idx, r in enumerate(all_face_records):
        emb_list = r.embedding
        assert isinstance(emb_list, list), f"Record #{r.id} embedding must be a list"
        assert len(emb_list) == 128, f"Record #{r.id} embedding dimension must be 128, got {len(emb_list)}"
        arr = np.array(emb_list, dtype=np.float32)
        assert np.all(np.isfinite(arr)), f"Record #{r.id} contains non-finite numbers"
        norm = float(np.linalg.norm(arr))
        assert abs(norm - 1.0) < 1e-3, f"Record #{r.id} L2 norm not 1.0 (got {norm})"
        print(f"  Record #{r.id}: User #{r.user_id} | Dim={len(emb_list)} | Finite={bool(np.all(np.isfinite(arr)))} | L2 Norm={norm:.6f} | Created={r.created_at}")

    # -------------------------------------------------------------
    # 9. INTEGRITY CHECK: Existing Iris & Access Log Records
    # -------------------------------------------------------------
    print(f"\n[TEST 9] Integrity Check of Existing Database Tables:")
    iris_v2_count_after = db.query(IrisEmbeddingV2).count()
    iris_v1_count_after = db.query(IrisEmbedding).count()
    access_logs_count_after = db.query(AccessLog).count()
    users_count_after = db.query(User).count()

    print(f"  Users: before={len(users)}, after={users_count_after} (Diff={users_count_after - len(users)})")
    print(f"  Iris V2 Embeddings: before={iris_v2_count_before}, after={iris_v2_count_after} (Diff={iris_v2_count_after - iris_v2_count_before})")
    print(f"  Iris V1 Embeddings: before={iris_v1_count_before}, after={iris_v1_count_after} (Diff={iris_v1_count_after - iris_v1_count_before})")
    print(f"  Access Logs: before={access_logs_count_before}, after={access_logs_count_after} (Diff={access_logs_count_after - access_logs_count_before})")

    assert iris_v2_count_before == iris_v2_count_after, "Iris V2 embeddings changed!"
    assert iris_v1_count_before == iris_v1_count_after, "Iris V1 embeddings changed!"
    assert access_logs_count_before == access_logs_count_after, "Access logs changed!"
    assert len(users) == users_count_after, "Users table changed!"

    print("\n" + "=" * 70)
    print("ALL STEP 2 TESTS COMPLETED AND VERIFIED WITH 100% SUCCESS!")
    print("=" * 70)

finally:
    db.close()
