"""
Face Recognition Service
========================
Provides face detection using YuNet ONNX model and 128-dimensional
feature embedding extraction using SFace ONNX model via OpenCV DNN.

Models:
- YuNet: ai_models/trained_models/face_detection_yunet_2023mar.onnx
- SFace: ai_models/trained_models/face_recognition_sface_2021dec.onnx

Features:
- Thread-safe singleton model caching (loaded once, reused for all requests)
- Strict single-face requirement enforcement
- L2-normalized 128-D embedding output suitable for cosine similarity
- Comprehensive error handling for no-face, multi-face, and invalid inputs
"""

import os
import io
import threading
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
from PIL import Image
from datetime import datetime

# ---------------------------------------------------------------------------
# Path Configuration & Model Resolvers
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

YUNET_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models",
    "trained_models",
    "face_detection_yunet_2023mar.onnx"
)

SFACE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models",
    "trained_models",
    "face_recognition_sface_2021dec.onnx"
)

# ---------------------------------------------------------------------------
# Thread-Safe Singleton Instances
# ---------------------------------------------------------------------------
_detector_lock = threading.Lock()
_recognizer_lock = threading.Lock()

_detector_instance: Optional[cv2.FaceDetectorYN] = None
_recognizer_instance: Optional[cv2.FaceRecognizerSF] = None


def get_face_detector(
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000
) -> cv2.FaceDetectorYN:
    """
    Returns the cached singleton instance of cv2.FaceDetectorYN (YuNet).
    Loads the model from disk once and reuses it across calls.
    """
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                if not os.path.exists(YUNET_MODEL_PATH):
                    raise FileNotFoundError(
                        f"YuNet ONNX model not found at: {YUNET_MODEL_PATH}"
                    )
                _detector_instance = cv2.FaceDetectorYN.create(
                    model=YUNET_MODEL_PATH,
                    config="",
                    input_size=(320, 320),
                    score_threshold=score_threshold,
                    nms_threshold=nms_threshold,
                    top_k=top_k
                )
    return _detector_instance


def get_face_recognizer() -> cv2.FaceRecognizerSF:
    """
    Returns the cached singleton instance of cv2.FaceRecognizerSF (SFace).
    Loads the model from disk once and reuses it across calls.
    """
    global _recognizer_instance
    if _recognizer_instance is None:
        with _recognizer_lock:
            if _recognizer_instance is None:
                if not os.path.exists(SFACE_MODEL_PATH):
                    raise FileNotFoundError(
                        f"SFace ONNX model not found at: {SFACE_MODEL_PATH}"
                    )
                _recognizer_instance = cv2.FaceRecognizerSF.create(
                    model=SFACE_MODEL_PATH,
                    config=""
                )
    return _recognizer_instance


# ---------------------------------------------------------------------------
# Image Preprocessing & Input Validation
# ---------------------------------------------------------------------------
def _coerce_to_bgr(image: Any) -> Optional[np.ndarray]:
    """
    Converts various input formats (numpy array, PIL Image, bytes, filepath)
    into a valid uint8 3-channel OpenCV BGR image array.
    """
    if image is None:
        return None

    # Handle file path string
    if isinstance(image, (str, os.PathLike)):
        if os.path.exists(image):
            img = cv2.imread(str(image))
            return img
        return None

    # Handle raw bytes
    if isinstance(image, bytes):
        arr = np.frombuffer(image, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img

    # Handle PIL Image
    if isinstance(image, Image.Image):
        rgb_arr = np.array(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    # Handle numpy ndarray
    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None

        # Grayscale (H, W) -> BGR (H, W, 3)
        if len(image.shape) == 2:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # 3-channel (H, W, 3)
        if len(image.shape) == 3 and image.shape[2] == 3:
            return image.astype(np.uint8)

        # 4-channel BGRA/RGBA (H, W, 4) -> BGR (H, W, 3)
        if len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGRA2BGR)

    return None


# ---------------------------------------------------------------------------
# Core Face Embedding Extraction Function
# ---------------------------------------------------------------------------
def extract_face_embedding(
    image: Any,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    require_single_face: bool = True
) -> Dict[str, Any]:
    """
    Detects a face in the input image using YuNet and extracts a 128-D
    L2-normalized feature embedding using SFace.

    Parameters:
    -----------
    image:
        OpenCV BGR numpy array, PIL Image, image bytes, or image filepath.
    score_threshold: float (default 0.6)
        Minimum YuNet confidence score required for face candidate detection.
    nms_threshold: float (default 0.3)
        Non-maximum suppression threshold for YuNet.
    require_single_face: bool (default True)
        If True, rejects images containing zero or multiple faces to guarantee
        unambiguous 1-person identity verification.

    Returns:
    --------
    dict with keys:
        - "status": str ("SUCCESS", "NO_FACE_DETECTED", "MULTIPLE_FACES_DETECTED",
                         "INVALID_IMAGE", "MODEL_LOAD_ERROR")
        - "success": bool (True on valid single-face extraction)
        - "message": str (Human-readable description)
        - "embedding": List[float] (128-D L2-normalized vector) or None
        - "embedding_dim": int (128 on success, 0 on failure)
        - "bbox": dict {"x": int, "y": int, "w": int, "h": int} or None
        - "confidence": float (YuNet score) or 0.0
        - "num_faces": int (Count of faces detected)
        - "landmarks": dict with 5 facial keypoints or None
        - "all_bboxes": List[dict] (Bounding boxes for all detected faces)
    """
    # 1. Coerce input image to BGR ndarray
    bgr_img = _coerce_to_bgr(image)
    if bgr_img is None or bgr_img.size == 0:
        return {
            "status": "INVALID_IMAGE",
            "success": False,
            "message": "Input image is invalid, empty, or unreadable.",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": None,
            "confidence": 0.0,
            "num_faces": 0,
            "landmarks": None,
            "all_bboxes": []
        }

    h, w = bgr_img.shape[:2]
    if h < 20 or w < 20:
        return {
            "status": "INVALID_IMAGE",
            "success": False,
            "message": f"Input image dimensions too small ({w}x{h}).",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": None,
            "confidence": 0.0,
            "num_faces": 0,
            "landmarks": None,
            "all_bboxes": []
        }

    # 2. Acquire model instances
    try:
        detector = get_face_detector(
            score_threshold=score_threshold,
            nms_threshold=nms_threshold
        )
        recognizer = get_face_recognizer()
    except Exception as e:
        return {
            "status": "MODEL_LOAD_ERROR",
            "success": False,
            "message": f"Failed to initialize YuNet/SFace models: {str(e)}",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": None,
            "confidence": 0.0,
            "num_faces": 0,
            "landmarks": None,
            "all_bboxes": []
        }

    # 3. Detect faces with YuNet
    with _detector_lock:
        detector.setInputSize((w, h))
        _, raw_faces = detector.detect(bgr_img)

    detected_faces = []
    if raw_faces is not None and len(raw_faces) > 0:
        detected_faces = list(raw_faces)

    num_faces = len(detected_faces)

    # Format all detected bounding boxes
    all_bboxes = []
    for f in detected_faces:
        bx, by, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        b_conf = float(f[-1])
        all_bboxes.append({"x": bx, "y": by, "w": bw, "h": bh, "confidence": b_conf})

    # 4. Handle no-face detection
    if num_faces == 0:
        return {
            "status": "NO_FACE_DETECTED",
            "success": False,
            "message": "No face detected in the image.",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": None,
            "confidence": 0.0,
            "num_faces": 0,
            "landmarks": None,
            "all_bboxes": []
        }

    # 5. Handle multiple-faces detection (if strict single-face required)
    if require_single_face and num_faces > 1:
        return {
            "status": "MULTIPLE_FACES_DETECTED",
            "success": False,
            "message": f"Multiple faces detected ({num_faces} faces). Exactly one person must be present.",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": None,
            "confidence": float(detected_faces[0][-1]),
            "num_faces": num_faces,
            "landmarks": None,
            "all_bboxes": all_bboxes
        }

    # 6. Extract primary face candidate
    face_candidate = detected_faces[0]
    fx, fy, fw, fh = int(face_candidate[0]), int(face_candidate[1]), int(face_candidate[2]), int(face_candidate[3])
    confidence = float(face_candidate[-1])

    landmarks = {
        "right_eye": [float(face_candidate[4]), float(face_candidate[5])],
        "left_eye": [float(face_candidate[6]), float(face_candidate[7])],
        "nose": [float(face_candidate[8]), float(face_candidate[9])],
        "right_mouth": [float(face_candidate[10]), float(face_candidate[11])],
        "left_mouth": [float(face_candidate[12]), float(face_candidate[13])]
    }

    # 7. Crop & Align face for SFace, then extract feature embedding
    with _recognizer_lock:
        aligned_face = recognizer.alignCrop(bgr_img, face_candidate)
        raw_feat = recognizer.feature(aligned_face)

    # 8. Consistent L2 Normalization (norm = 1.0)
    feat_arr = np.array(raw_feat, dtype=np.float32).flatten()
    l2_norm = float(np.linalg.norm(feat_arr))

    if l2_norm > 1e-7:
        normalized_feat = feat_arr / l2_norm
    else:
        normalized_feat = feat_arr

    # Validate numbers are finite
    if not np.all(np.isfinite(normalized_feat)):
        return {
            "status": "EXTRACTION_ERROR",
            "success": False,
            "message": "Generated embedding vector contains non-finite (NaN/Inf) values.",
            "embedding": None,
            "embedding_dim": 0,
            "bbox": {"x": fx, "y": fy, "w": fw, "h": fh},
            "confidence": confidence,
            "num_faces": num_faces,
            "landmarks": landmarks,
            "all_bboxes": all_bboxes
        }

    embedding_list = [float(v) for v in normalized_feat.tolist()]

    return {
        "status": "SUCCESS",
        "success": True,
        "message": "Face embedding extracted successfully.",
        "embedding": embedding_list,
        "embedding_dim": len(embedding_list),
        "bbox": {"x": fx, "y": fy, "w": fw, "h": fh},
        "confidence": confidence,
        "num_faces": num_faces,
        "landmarks": landmarks,
        "all_bboxes": all_bboxes
    }


# ---------------------------------------------------------------------------
# Cosine Similarity Calculation Helper
# ---------------------------------------------------------------------------
def compute_face_similarity(
    embedding1: Union[List[float], np.ndarray],
    embedding2: Union[List[float], np.ndarray]
) -> float:
    """
    Computes cosine similarity between two 128-D face embeddings.
    Since embeddings are L2-normalized, cosine similarity equals the dot product.

    Returns:
    --------
    float in range [-1.0, 1.0] (typically > 0.36 for the same individual with SFace).
    """
    if embedding1 is None or embedding2 is None:
        return 0.0

    v1 = np.array(embedding1, dtype=np.float32).flatten()
    v2 = np.array(embedding2, dtype=np.float32).flatten()

    if len(v1) != 128 or len(v2) != 128:
        return 0.0

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-7 or n2 < 1e-7:
        return 0.0

    # Ensure normalized before dot product
    v1_norm = v1 / n1
    v2_norm = v2 / n2

    sim = float(np.dot(v1_norm, v2_norm))
    # Clamp to valid [-1.0, 1.0] range
    return max(-1.0, min(1.0, sim))


# ---------------------------------------------------------------------------
# Database Face Sample Registration & Storage
# ---------------------------------------------------------------------------
def register_face_sample(
    user_id: int,
    image_data: Any,
    db: Any = None,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Registers a single face embedding sample for an existing user in MySQL:
    1. Verifies that the user exists.
    2. Extracts a 128-D L2-normalized embedding using YuNet + SFace.
    3. Rejects on no-face, multiple-faces, or invalid image.
    4. Validates 128-D length, finite numbers, and L2 normalization.
    5. Inserts a new FaceEmbedding record (allows multiple samples per user, never overwriting).
    """
    should_close = False
    if db is None:
        from app.database.connection import SessionLocal
        db = SessionLocal()
        should_close = True

    try:
        from app.models.user_model import User, FaceEmbedding

        # 1. Verify user exists
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {
                "status": "error",
                "error_code": "USER_NOT_FOUND",
                "message": f"User ID #{user_id} does not exist in the database.",
                "user_id": user_id,
                "sample_id": None,
                "total_face_samples": 0
            }

        # 2. Extract face embedding
        extract_res = extract_face_embedding(
            image=image_data,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            require_single_face=True
        )

        if not extract_res.get("success"):
            return {
                "status": "error",
                "error_code": extract_res.get("status", "EXTRACTION_FAILED"),
                "message": extract_res.get("message", "Face extraction failed."),
                "user_id": user_id,
                "sample_id": None,
                "num_faces": extract_res.get("num_faces", 0),
                "confidence": extract_res.get("confidence", 0.0),
                "total_face_samples": db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).count()
            }

        embedding = extract_res["embedding"]

        # 3. Strict Validation Before Storing:
        # - Exactly 128 values
        # - All values finite
        # - Embedding is L2-normalized (~1.0)
        if not isinstance(embedding, list) or len(embedding) != 128:
            return {
                "status": "error",
                "error_code": "INVALID_EMBEDDING_DIMENSION",
                "message": f"Embedding dimension must be exactly 128, got {len(embedding) if isinstance(embedding, list) else 0}",
                "user_id": user_id,
                "sample_id": None
            }

        emb_arr = np.array(embedding, dtype=np.float32)
        if not np.all(np.isfinite(emb_arr)):
            return {
                "status": "error",
                "error_code": "NON_FINITE_VALUES",
                "message": "Embedding contains non-finite values (NaN / Inf).",
                "user_id": user_id,
                "sample_id": None
            }

        l2_norm = float(np.linalg.norm(emb_arr))
        if abs(l2_norm - 1.0) > 1e-3:
            return {
                "status": "error",
                "error_code": "INVALID_NORMALIZATION",
                "message": f"Embedding must be L2-normalized (expected 1.0, got {l2_norm:.6f}).",
                "user_id": user_id,
                "sample_id": None
            }

        # 4. Insert new FaceEmbedding record (multiple samples allowed, never overwriting)
        new_record = FaceEmbedding(
            user_id=user.id,
            embedding=embedding
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)

        total_samples = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).count()

        return {
            "status": "success",
            "message": f"Face sample successfully registered for user '{user.name}' (#{user.id}).",
            "user_id": user.id,
            "user_name": user.name,
            "sample_id": new_record.id,
            "total_face_samples": total_samples,
            "confidence": extract_res["confidence"],
            "bbox": extract_res["bbox"],
            "landmarks": extract_res["landmarks"],
            "embedding_dim": 128,
            "l2_norm": l2_norm,
            "created_at": new_record.created_at.isoformat() if new_record.created_at else None
        }
    except Exception as e:
        if db:
            db.rollback()
        return {
            "status": "error",
            "error_code": "DATABASE_ERROR",
            "message": f"Failed to persist face embedding: {str(e)}",
            "user_id": user_id,
            "sample_id": None
        }
    finally:
        if should_close and db:
            db.close()


def get_user_face_samples(
    user_id: int,
    db: Any = None
) -> Dict[str, Any]:
    """
    Retrieves all registered face embedding sample metadata for a user.
    """
    should_close = False
    if db is None:
        from app.database.connection import SessionLocal
        db = SessionLocal()
        should_close = True

    try:
        from app.models.user_model import User, FaceEmbedding

        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {
                "status": "error",
                "message": f"User ID #{user_id} not found.",
                "samples": []
            }

        records = db.query(FaceEmbedding).filter(FaceEmbedding.user_id == user.id).order_by(FaceEmbedding.id.asc()).all()
        samples = []
        for r in records:
            samples.append({
                "id": r.id,
                "user_id": r.user_id,
                "embedding_dim": len(r.embedding) if isinstance(r.embedding, list) else 0,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })

        return {
            "status": "success",
            "user_id": user.id,
            "user_name": user.name,
            "total_samples": len(samples),
            "samples": samples
        }
    finally:
        if should_close and db:
            db.close()


# ---------------------------------------------------------------------------
# Face Matching Threshold Configuration
# ---------------------------------------------------------------------------
# NOTE: OpenCV SFace benchmark reference threshold for cosine similarity
# is approximately 0.363 (LFW verification benchmark).
#
# MARKED AS DEVELOPMENT ONLY:
# This threshold is configurable via the FACE_MATCH_THRESHOLD environment
# variable and is intended for development, testing, and candidate selection.
# It is NOT claimed to be a final production-validated security threshold.
# The existing iris threshold remains completely unchanged and separate.
DEFAULT_FACE_MATCH_THRESHOLD: float = float(
    os.getenv("FACE_MATCH_THRESHOLD", "0.360")
)


# ---------------------------------------------------------------------------
# Stage 1: 1:N Facial Identity Matching
# ---------------------------------------------------------------------------
def identify_user_by_face(
    image: Any,
    db: Any = None,
    threshold: Optional[float] = None,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Stage 1: 1:N Facial Identity Matching.
    Given an input image (OpenCV BGR array, PIL Image, bytes, or file path):
    1. Detects face using YuNet (enforces exactly 1 face).
    2. Extracts 128-D SFace L2-normalized embedding.
    3. Queries enrolled face embeddings from MySQL `face_embeddings`.
    4. Compares live embedding against each enrolled embedding via cosine similarity.
    5. Aggregates scores per user using the maximum similarity for that user:
       user_score = max(sample_scores_for_user)
    6. Selects the top matching user.
    7. If top score >= FACE_MATCH_THRESHOLD:
       returns candidate_user_id and matched=True.
       Else:
       returns matched=False and reason="IDENTITY_NOT_FOUND".
    """
    match_thresh = threshold if threshold is not None else DEFAULT_FACE_MATCH_THRESHOLD

    # 1. Extract live face embedding
    extract_res = extract_face_embedding(
        image=image,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
        require_single_face=True
    )

    if not extract_res.get("success"):
        err_status = extract_res.get("status", "EXTRACTION_FAILED")
        return {
            "status": "error",
            "error_code": err_status,
            "matched": False,
            "reason": err_status,
            "message": extract_res.get("message", "Face detection/extraction failed."),
            "candidate_user_id": None,
            "candidate_user_name": None,
            "face_score": 0.0,
            "threshold": match_thresh,
            "confidence": extract_res.get("confidence", 0.0),
            "bbox": extract_res.get("bbox"),
            "landmarks": extract_res.get("landmarks"),
            "num_faces": extract_res.get("num_faces", 0),
            "all_bboxes": extract_res.get("all_bboxes", []),
            "user_scores": {},
            "is_development_threshold": True
        }

    live_emb = extract_res["embedding"]

    # 2. Query enrolled face embeddings from database
    should_close = False
    if db is None:
        from app.database.connection import SessionLocal
        db = SessionLocal()
        should_close = True

    try:
        from app.models.user_model import User, FaceEmbedding

        records = (
            db.query(FaceEmbedding, User.name)
            .join(User, FaceEmbedding.user_id == User.id)
            .all()
        )

        if not records:
            return {
                "status": "success",
                "matched": False,
                "reason": "NO_REGISTERED_USERS",
                "message": "No registered face embeddings exist in the MySQL database.",
                "candidate_user_id": None,
                "candidate_user_name": None,
                "face_score": 0.0,
                "threshold": match_thresh,
                "confidence": extract_res["confidence"],
                "bbox": extract_res["bbox"],
                "landmarks": extract_res["landmarks"],
                "num_faces": 1,
                "user_scores": {},
                "is_development_threshold": True
            }

        # 3. Compare at embedding level and aggregate per user
        # user_scores structure: { user_id: { "user_name": str, "sample_scores": [float, ...], "best_score": float } }
        user_scores: Dict[int, Dict[str, Any]] = {}
        for face_rec, user_name in records:
            uid = face_rec.user_id
            if uid not in user_scores:
                user_scores[uid] = {
                    "user_name": user_name,
                    "sample_scores": [],
                    "sample_ids": [],
                    "best_score": -1.0
                }

            sim = compute_face_similarity(live_emb, face_rec.embedding)
            sim_rounded = round(sim, 4)
            user_scores[uid]["sample_scores"].append(sim_rounded)
            user_scores[uid]["sample_ids"].append(face_rec.id)

        # Compute best_score per user (max across their registered samples)
        for uid in user_scores:
            if user_scores[uid]["sample_scores"]:
                user_scores[uid]["best_score"] = max(user_scores[uid]["sample_scores"])

        # 4. Determine top matching user
        sorted_users = sorted(
            user_scores.items(),
            key=lambda item: item[1]["best_score"],
            reverse=True
        )
        top_user_id, top_user_info = sorted_users[0]
        best_score = top_user_info["best_score"]
        top_user_name = top_user_info["user_name"]

        # 5. Evaluate threshold
        if best_score >= match_thresh:
            return {
                "status": "success",
                "matched": True,
                "reason": "IDENTITY_VERIFIED",
                "message": f"Face candidate identified: '{top_user_name}' (#{top_user_id}) with score {best_score:.4f} (>= {match_thresh:.3f}).",
                "candidate_user_id": top_user_id,
                "candidate_user_name": top_user_name,
                "face_score": best_score,
                "threshold": match_thresh,
                "confidence": extract_res["confidence"],
                "bbox": extract_res["bbox"],
                "landmarks": extract_res["landmarks"],
                "num_faces": 1,
                "user_scores": {str(k): v for k, v in user_scores.items()},
                "is_development_threshold": True
            }
        else:
            return {
                "status": "success",
                "matched": False,
                "reason": "IDENTITY_NOT_FOUND",
                "message": f"No registered face passed threshold. Best score was {best_score:.4f} for '{top_user_name}' (< {match_thresh:.3f}).",
                "candidate_user_id": None,
                "candidate_user_name": None,
                "face_score": best_score,
                "threshold": match_thresh,
                "confidence": extract_res["confidence"],
                "bbox": extract_res["bbox"],
                "landmarks": extract_res["landmarks"],
                "num_faces": 1,
                "user_scores": {str(k): v for k, v in user_scores.items()},
                "is_development_threshold": True
            }
    finally:
        if should_close and db:
            db.close()


def verify_two_stage_identity(
    face_image_input,
    iris_image_input=None,
    action: str = "CHECK-IN",
    dry_run: bool = False,
    db: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Two-Stage Identity Verification Engine:
    ======================================
    Stage 1: 1:N Facial Identity Matching (YuNet + SFace) -> selects candidate_user_id.
    Stage 2: 1:1 Biometric Iris Verification (Existing V2 model) against candidate_user_id templates ONLY.

    Safety:
    - If Stage 1 fails (IDENTITY_NOT_FOUND, NO_FACE_DETECTED, MULTIPLE_FACES_DETECTED),
      iris verification is bypassed entirely (executed = False).
    - If dry_run is True, hardware commands (A/U to COM9) and access_logs mutations are suppressed.
    - Final Decision: VERIFIED only when Face MATCH + Iris MATCH. Everything else is REJECTED.
    """
    # 1. Execute Stage 1: Face 1:N Identification
    face_result = identify_user_by_face(face_image_input, db=db)

    # Stage 1 Failure Handling
    if not face_result.get("matched", False):
        return {
            "face_stage": {
                "matched": False,
                "status": face_result.get("status"),
                "reason": face_result.get("reason"),
                "candidate_user_id": None,
                "candidate_user_name": None,
                "score": round(float(face_result.get("face_score", 0.0)), 4),
                "threshold": face_result.get("threshold", DEFAULT_FACE_MATCH_THRESHOLD),
                "num_faces": face_result.get("num_faces", 0),
                "is_development_threshold": face_result.get("is_development_threshold", True)
            },
            "iris_stage": {
                "executed": False,
                "user_id_checked": None,
                "score": 0.0,
                "threshold": 0.80,
                "matched": False,
                "template_ids_used": [],
                "templates_checked": 0
            },
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if dry_run else "NONE"
        }

    # Stage 1 Succeeded -> Candidate user selected
    candidate_user_id = face_result.get("candidate_user_id")
    candidate_user_name = face_result.get("candidate_user_name")
    face_score = face_result.get("face_score", 0.0)

    # 2. Execute Stage 2: Iris Verification ONLY against candidate_user_id templates
    target_iris = iris_image_input if iris_image_input is not None else face_image_input

    # Lazy import to avoid circular dependencies
    from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic

    iris_result = recognize_iris_v2_1to1_diagnostic(
        target_iris,
        claimed_user_id=candidate_user_id,
        action=action,
        candidate_only=True,
        dry_run=dry_run
    )

    action_verified = (iris_result.get("final_diagnostic_status") == "CLAIMED_IDENTITY_VERIFIED")
    iris_matched = bool(iris_result.get("biometric_verified", False)) and action_verified
    iris_score = float(iris_result.get("best_similarity", 0.0))
    iris_threshold = float(iris_result.get("diagnostic_threshold", 0.80))
    template_ids_used = iris_result.get("template_ids_used", [])

    # 3. Final Decision Logic
    # Case 3: Face MATCH + Iris MATCH + Action VALID -> VERIFIED
    # Case 2: Face MATCH + Iris FAIL or Action INVALID -> REJECTED
    if iris_matched:
        final_decision = "VERIFIED"
        failure_stage = None
    else:
        final_decision = "REJECTED"
        failure_stage = "ACCESS_CONTROL" if (iris_result.get("biometric_verified") and not action_verified) else "BIOMETRIC_VERIFICATION"

    return {
        "face_stage": {
            "matched": True,
            "candidate_user_id": candidate_user_id,
            "candidate_user_name": candidate_user_name,
            "score": round(float(face_score), 4),
            "threshold": face_result.get("threshold", DEFAULT_FACE_MATCH_THRESHOLD),
            "is_development_threshold": face_result.get("is_development_threshold", True)
        },
        "iris_stage": {
            "executed": True,
            "user_id_checked": candidate_user_id,
            "left_score": iris_result.get("left_score"),
            "right_score": iris_result.get("right_score"),
            "score": round(iris_score, 4),
            "threshold": iris_threshold,
            "matched": iris_matched,
            "template_ids_used": template_ids_used,
            "templates_checked": len(template_ids_used),
            "decision_reason": iris_result.get("decision_reason"),
            "final_diagnostic_status": iris_result.get("final_diagnostic_status")
        },
        "final_decision": final_decision,
        "failure_stage": failure_stage,
        "check_in_time": iris_result.get("check_in_time"),
        "check_in_time_formatted": iris_result.get("check_in_time_formatted"),
        "check_out_time": iris_result.get("check_out_time"),
        "check_out_time_formatted": iris_result.get("check_out_time_formatted"),
        "total_duration": iris_result.get("total_duration"),
        "is_inside": iris_result.get("is_inside", (action == "CHECK-IN" and final_decision == "VERIFIED")),
        "decision_reason": iris_result.get("decision_reason"),
        "hardware_action": "SUPPRESSED_DIAGNOSTIC"
    }


def verify_two_stage_sequence_identity(
    frames_input: List[Any],
    action: str = "CHECK-IN",
    dry_run: bool = True,
    db: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Coordinated Multi-Frame Two-Stage Identity Verification:
    Stage 1: 1:N Facial Identity Matching across captured sequence.
             Evaluates primary frontal frame first (index 0); falls back to other frames if needed.
    Stage 2: 1:1 Biometric Iris Verification against identified candidate's templates.
             Evaluates candidate's iris biometrics across the frames to track best score.
             Evaluates strict threshold (0.800).
    Safety:
    - If dry_run is True, physical access is not granted and hardware commands are suppressed.
    - Thresholds are strictly maintained: face (0.360), iris (0.800).
    """
    if not frames_input:
        return {
            "status": "error",
            "error_code": "NO_FRAMES_PROVIDED",
            "face_stage": {"matched": False},
            "iris_stage": {"executed": False},
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC"
        }

    if len(frames_input) == 1:
        return verify_two_stage_identity(
            face_image_input=frames_input[0],
            action=action,
            dry_run=dry_run,
            db=db
        )

    # Stage 1: Face Candidate Identification (1:N)
    # Check Frame 0 (frontal pose) first
    face_result = identify_user_by_face(frames_input[0], db=db)
    if not face_result.get("matched", False):
        # Fallback to remaining frames if frame 0 failed
        for f_idx in range(1, len(frames_input)):
            cand_res = identify_user_by_face(frames_input[f_idx], db=db)
            if cand_res.get("matched", False):
                face_result = cand_res
                break

    if not face_result.get("matched", False):
        return {
            "face_stage": {
                "matched": False,
                "status": face_result.get("status"),
                "reason": face_result.get("reason"),
                "candidate_user_id": None,
                "candidate_user_name": None,
                "score": round(float(face_result.get("face_score", 0.0)), 4),
                "threshold": face_result.get("threshold", DEFAULT_FACE_MATCH_THRESHOLD),
                "num_faces": face_result.get("num_faces", 0),
                "is_development_threshold": face_result.get("is_development_threshold", True)
            },
            "iris_stage": {
                "executed": False,
                "user_id_checked": None,
                "score": 0.0,
                "threshold": 0.80,
                "matched": False,
                "template_ids_used": [],
                "templates_checked": 0
            },
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if dry_run else "NO_COMMAND_IDENTITY_NOT_FOUND",
            "frames_evaluated": len(frames_input)
        }

    candidate_user_id = face_result.get("candidate_user_id")
    candidate_user_name = face_result.get("candidate_user_name")
    face_score = face_result.get("face_score", 0.0)

    # Stage 2: Iris Verification against candidate's enrolled templates
    from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic, is_user_currently_inside, log_access_event
    from app.database.connection import SessionLocal

    best_iris_result = None
    best_score = -1.0

    for frame in frames_input:
        iris_res = recognize_iris_v2_1to1_diagnostic(
            frame,
            claimed_user_id=candidate_user_id,
            action=action,
            candidate_only=True,
            dry_run=True
        )
        score = float(iris_res.get("best_similarity", 0.0))
        if score > best_score:
            best_score = score
            best_iris_result = iris_res

        # If verified, we found a successful frame
        if iris_res.get("final_diagnostic_status") == "CLAIMED_IDENTITY_VERIFIED" or iris_res.get("biometric_verified"):
            best_iris_result = iris_res
            best_score = score
            break

    if best_iris_result is None:
        best_iris_result = recognize_iris_v2_1to1_diagnostic(
            frames_input[0],
            claimed_user_id=candidate_user_id,
            action=action,
            candidate_only=True,
            dry_run=True
        )

    biometrics_passed = bool(best_iris_result.get("biometric_verified", False))
    iris_score = float(best_iris_result.get("best_similarity", 0.0))
    iris_threshold = float(best_iris_result.get("diagnostic_threshold", 0.80))
    template_ids_used = best_iris_result.get("template_ids_used", [])

    check_in_time_iso = None
    check_in_time_fmt = None
    check_out_time_iso = None
    check_out_time_fmt = None
    total_duration_str = None
    user_is_inside = False

    db_log = SessionLocal()
    try:
        if biometrics_passed:
            is_inside, last_in = is_user_currently_inside(db_log, candidate_user_name)
            if action == "CHECK-IN":
                if is_inside:
                    # User is ALREADY inside -> Reject duplicate check-in, do not mutate DB, do not sound alarm
                    final_decision = "REJECTED"
                    failure_stage = "ACCESS_CONTROL"
                    decision_reason = "USER_ALREADY_INSIDE"
                    user_is_inside = True
                    if last_in and last_in.timestamp:
                        check_in_time_iso = last_in.timestamp.isoformat()
                        check_in_time_fmt = last_in.timestamp.strftime("%I:%M:%S %p")
                else:
                    final_decision = "VERIFIED"
                    failure_stage = None
                    decision_reason = "CHECK-IN VERIFIED"
                    user_is_inside = True
                    saved_log = None
                    if not dry_run:
                        saved_log = log_access_event(db_log, candidate_user_name, "ACCESS_GRANTED", iris_score, action="CHECK-IN")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_in_time_iso = now_dt.isoformat()
                    check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")
            elif action == "CHECK-OUT":
                if not is_inside:
                    # User is NOT inside -> Reject checkout
                    final_decision = "REJECTED"
                    failure_stage = "ACCESS_CONTROL"
                    decision_reason = "USER_NOT_INSIDE"
                    user_is_inside = False
                else:
                    final_decision = "VERIFIED"
                    failure_stage = None
                    decision_reason = "CHECK-OUT VERIFIED"
                    user_is_inside = False
                    saved_log = None
                    if not dry_run:
                        saved_log = log_access_event(db_log, candidate_user_name, "ACCESS_GRANTED", iris_score, action="CHECK-OUT")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_out_time_iso = now_dt.isoformat()
                    check_out_time_fmt = now_dt.strftime("%I:%M:%S %p")

                    if last_in and last_in.timestamp:
                        check_in_time_iso = last_in.timestamp.isoformat()
                        check_in_time_fmt = last_in.timestamp.strftime("%I:%M:%S %p")
                        diff_sec = max(0, int((now_dt - last_in.timestamp).total_seconds()))
                        hrs = diff_sec // 3600
                        mins = (diff_sec % 3600) // 60
                        secs = diff_sec % 60
                        total_duration_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"
                    else:
                        check_in_time_iso = now_dt.isoformat()
                        check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")
                        total_duration_str = "00h 00m 00s"
            else:
                final_decision = "VERIFIED"
                failure_stage = None
                decision_reason = "DIAGNOSTIC VERIFIED"
                if not dry_run:
                    log_access_event(db_log, candidate_user_name, "ACCESS_GRANTED", iris_score, action=action)
        else:
            final_decision = "REJECTED"
            failure_stage = "BIOMETRIC_VERIFICATION"
            decision_reason = best_iris_result.get("decision_reason", "BIOMETRIC_MISMATCH")
            if not dry_run:
                log_access_event(db_log, candidate_user_name, "ACCESS_DENIED", iris_score if iris_score > 0 else 0.0, action=action)
    finally:
        db_log.close()

    if dry_run:
        hw_action = "SUPPRESSED_DIAGNOSTIC"
    elif final_decision == "VERIFIED":
        hw_action = "SENT_A_AUTHORIZED"
    elif failure_stage == "BIOMETRIC_VERIFICATION":
        hw_action = "SENT_U_UNAUTHORIZED"
    else:
        hw_action = f"NO_COMMAND_{failure_stage}"

    return {
        "face_stage": {
            "matched": True,
            "candidate_user_id": candidate_user_id,
            "candidate_user_name": candidate_user_name,
            "score": round(float(face_score), 4),
            "threshold": face_result.get("threshold", DEFAULT_FACE_MATCH_THRESHOLD),
            "is_development_threshold": face_result.get("is_development_threshold", True)
        },
        "iris_stage": {
            "executed": True,
            "user_id_checked": candidate_user_id,
            "left_score": best_iris_result.get("left_score"),
            "right_score": best_iris_result.get("right_score"),
            "score": round(iris_score, 4),
            "threshold": iris_threshold,
            "matched": (final_decision == "VERIFIED"),
            "template_ids_used": template_ids_used,
            "templates_checked": len(template_ids_used),
            "decision_reason": decision_reason,
            "final_diagnostic_status": "CLAIMED_IDENTITY_VERIFIED" if final_decision == "VERIFIED" else "CLAIMED_IDENTITY_NOT_VERIFIED"
        },
        "final_decision": final_decision,
        "failure_stage": failure_stage,
        "decision_reason": decision_reason,
        "check_in_time": check_in_time_iso,
        "check_in_time_formatted": check_in_time_fmt,
        "check_out_time": check_out_time_iso,
        "check_out_time_formatted": check_out_time_fmt,
        "total_duration": total_duration_str,
        "is_inside": user_is_inside,
        "hardware_action": hw_action,
        "frames_evaluated": len(frames_input)
    }



