import sys
import os
import cv2
import numpy as np

# Add backend directory to sys.path
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend"))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.services.face_recognition_service import (
    extract_face_embedding,
    compute_face_similarity,
    get_face_detector,
    get_face_recognizer
)

print("=" * 65)
print("TEST SUITE: FACE RECOGNITION SERVICE (YuNet + SFace)")
print("=" * 65)

# --- Test 1: Model Loading & Singleton Verification ---
print("\n[TEST 1] Model Loading & Singleton Verification:")
det1 = get_face_detector()
det2 = get_face_detector()
rec1 = get_face_recognizer()
rec2 = get_face_recognizer()

print(f"  YuNet detector loaded successfully: {det1 is not None} (id={id(det1)})")
print(f"  YuNet singleton reused across calls: {det1 is det2}")
print(f"  SFace recognizer loaded successfully: {rec1 is not None} (id={id(rec1)})")
print(f"  SFace singleton reused across calls: {rec1 is rec2}")

assert det1 is det2, "YuNet must be a cached singleton!"
assert rec1 is rec2, "SFace must be a cached singleton!"

# --- Test 2: Single-Face Image Test ---
print("\n[TEST 2] Single-Face Image Test:")
single_face_image_path = "project_images/003.jpeg"
print(f"  Target Image: {single_face_image_path}")
res_single = extract_face_embedding(single_face_image_path)

print(f"  Status: {res_single['status']}")
print(f"  Success: {res_single['success']}")
print(f"  Message: {res_single['message']}")
print(f"  Num Faces Detected: {res_single['num_faces']}")
print(f"  Bounding Box: {res_single['bbox']}")
print(f"  Confidence Score: {res_single['confidence']:.4f}")
print(f"  Landmarks: {list(res_single['landmarks'].keys())}")
print(f"  Embedding Dimension: {res_single['embedding_dim']}")

assert res_single["status"] == "SUCCESS", f"Expected SUCCESS, got {res_single['status']}"
assert res_single["num_faces"] == 1, f"Expected 1 face, got {res_single['num_faces']}"
assert res_single["embedding"] is not None, "Embedding should not be None"
assert res_single["embedding_dim"] == 128, f"Expected dim 128, got {res_single['embedding_dim']}"
assert len(res_single["embedding"]) == 128, f"Expected 128 elements, got {len(res_single['embedding'])}"

emb = np.array(res_single["embedding"], dtype=np.float32)
all_finite = bool(np.all(np.isfinite(emb)))
l2_norm = float(np.linalg.norm(emb))
print(f"  All values finite numbers: {all_finite}")
print(f"  L2 Norm: {l2_norm:.6f}")
print(f"  First 5 embedding values: {res_single['embedding'][:5]}")

assert all_finite, "All embedding values must be finite!"
assert abs(l2_norm - 1.0) < 1e-4, f"L2 norm should be 1.0, got {l2_norm}"

# --- Test 3: No-Face Image Test ---
print("\n[TEST 3] No-Face Image Test:")
no_face_image_path = "scratch/test_black.jpg"
print(f"  Target Image: {no_face_image_path}")
res_no_face = extract_face_embedding(no_face_image_path)

print(f"  Status: {res_no_face['status']}")
print(f"  Success: {res_no_face['success']}")
print(f"  Message: {res_no_face['message']}")
print(f"  Num Faces Detected: {res_no_face['num_faces']}")
print(f"  Embedding is None: {res_no_face['embedding'] is None}")

assert res_no_face["status"] == "NO_FACE_DETECTED", f"Expected NO_FACE_DETECTED, got {res_no_face['status']}"
assert res_no_face["num_faces"] == 0, f"Expected 0 faces, got {res_no_face['num_faces']}"
assert res_no_face["embedding"] is None, "Embedding must be None when no face detected"

# --- Test 4: Multiple-Faces Image Test ---
print("\n[TEST 4] Multiple-Faces Image Test:")
img_a = cv2.imread("project_images/003.jpeg")
img_b = cv2.imread("project_images/004.jpeg")
h_target = min(img_a.shape[0], img_b.shape[0])
img_a_scaled = cv2.resize(img_a, (int(img_a.shape[1] * h_target / img_a.shape[0]), h_target))
img_b_scaled = cv2.resize(img_b, (int(img_b.shape[1] * h_target / img_b.shape[0]), h_target))
multi_face_img = np.hstack([img_a_scaled, img_b_scaled])

res_multi = extract_face_embedding(multi_face_img, require_single_face=True)

print(f"  Status: {res_multi['status']}")
print(f"  Success: {res_multi['success']}")
print(f"  Message: {res_multi['message']}")
print(f"  Num Faces Detected: {res_multi['num_faces']}")
print(f"  All BBoxes Count: {len(res_multi['all_bboxes'])}")
print(f"  Embedding is None: {res_multi['embedding'] is None}")

assert res_multi["status"] == "MULTIPLE_FACES_DETECTED", f"Expected MULTIPLE_FACES_DETECTED, got {res_multi['status']}"
assert res_multi["num_faces"] == 2, f"Expected exactly 2 faces, got {res_multi['num_faces']}"
assert res_multi["embedding"] is None, "Embedding must be None when multiple faces detected"

# --- Test 5: Cosine Similarity Helper Test ---
print("\n[TEST 5] Cosine Similarity Verification:")
self_sim = compute_face_similarity(res_single["embedding"], res_single["embedding"])
print(f"  Self-Similarity (identical embedding): {self_sim:.6f}")
assert abs(self_sim - 1.0) < 1e-4, f"Self similarity should be 1.0, got {self_sim}"

res_second = extract_face_embedding("project_images/004.jpeg")
if res_second["success"]:
    diff_person_sim = compute_face_similarity(res_single["embedding"], res_second["embedding"])
    print(f"  Different Person Similarity (003 vs 004): {diff_person_sim:.4f}")
    assert -1.0 <= diff_person_sim <= 1.0, "Cosine similarity must be bounded in [-1.0, 1.0]"

# --- Test 6: Invalid Input Safety Test ---
print("\n[TEST 6] Invalid Input Safety Handling:")
res_none = extract_face_embedding(None)
print(f"  None input -> status: {res_none['status']}, success: {res_none['success']}")
assert res_none["status"] == "INVALID_IMAGE"

res_empty = extract_face_embedding(np.zeros((0, 0), dtype=np.uint8))
print(f"  Empty ndarray -> status: {res_empty['status']}, success: {res_empty['success']}")
assert res_empty["status"] == "INVALID_IMAGE"

res_tiny = extract_face_embedding(np.zeros((10, 10, 3), dtype=np.uint8))
print(f"  Tiny 10x10 ndarray -> status: {res_tiny['status']}, success: {res_tiny['success']}")
assert res_tiny["status"] == "INVALID_IMAGE"

print("\n" + "=" * 65)
print("ALL 6 TEST CASES PASSED PERFECTLY! ZERO ERRORS.")
print("=" * 65)
