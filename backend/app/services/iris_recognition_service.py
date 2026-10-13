import os
import io
import base64
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbedding, IrisEmbeddingV2, AccessLog
from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN
from app.services.iris_detection_service import detect_and_crop_iris
from app.services.anti_spoof_service import check_spoof
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/iris_feature_model.pth"
)

device = torch.device("cpu")

print("Initializing High-Precision Biometric Verification Engine with Strict 8-Step Decision Pipeline...")

recognition_model = IrisIdentityCNN(num_classes=2).to(device)

if os.path.exists(MODEL_PATH):
    recognition_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

recognition_model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

DEFAULT_IRIS_MATCH_THRESHOLD: float = float(os.getenv("IRIS_MATCH_THRESHOLD", "0.80"))
DEFAULT_IRIS_MIN_IDENTITY_MARGIN: float = float(os.getenv("IRIS_MIN_IDENTITY_MARGIN", "0.08"))


def log_access_event(db: Session, user_name: str, status: str, similarity_score: float, action: str = "CHECK-IN"):
    try:
        log_entry = AccessLog(
            user_name=user_name,
            status=status,
            action=action,
            similarity_score=round(float(similarity_score), 4)
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)
        return log_entry
    except Exception as e:
        print("Access Log Error:", e)
        return None


def is_user_currently_inside(db: Session, user_name: str):
    try:
        from sqlalchemy import func
        last_granted = (
            db.query(AccessLog)
            .filter(func.lower(AccessLog.user_name) == func.lower(user_name), AccessLog.status == "ACCESS_GRANTED")
            .order_by(AccessLog.timestamp.desc(), AccessLog.id.desc())
            .first()
        )
        if last_granted and (last_granted.action or "CHECK-IN").upper() == "CHECK-IN":
            return True, last_granted
        return False, None
    except Exception as e:
        print("Error checking inside status:", e)
        return False, None


def recognize_iris(image_input):
    """
    Mandatory 8-Step Biometric Decision Pipeline:
    STEP 1: Face ROI Detection -> face_detected
    STEP 2: Dual Eye & Pupil Localization -> left_eye_detected & right_eye_detected
    STEP 3: Liveness Anti-Spoofing -> liveness_passed
    STEP 4: Generate Query Feature Embedding -> query_embedding_generated
    STEP 5: Search ALL Enrolled Database Embeddings -> database_embeddings_checked
    STEP 6: Find GLOBAL BEST MATCH -> best_similarity, best_match_user_id, best_match_user_name
    STEP 7: Strict Match Threshold Evaluation (0.820)
            best_similarity >= threshold -> final_decision = ACCESS_GRANTED
            best_similarity <  threshold -> final_decision = ACCESS_DENIED
    STEP 8: Hardware Action & Diagnostic JSON Output
    """
    db: Session = SessionLocal()
    MATCH_THRESHOLD = 0.820

    try:
        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            pil_image = Image.open(image_input).convert("RGB")

        # STEP 1 & 2: Face and Dual Eye Localization
        polar_ribbon, annotated_img, face_detected, left_eye_detected, right_eye_detected = detect_and_crop_iris(pil_image)

        # Convert annotated image to Base64 data URL
        buffered = io.BytesIO()
        annotated_img.save(buffered, format="JPEG")
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        # Check STEP 1: Face Detection Guard
        if not face_detected:
            log_access_event(db, "NO_FACE_DETECTED", "ACCESS_DENIED", 0.0)
            return {
                "face_detected": False,
                "left_eye_detected": False,
                "right_eye_detected": False,
                "liveness_passed": False,
                "query_embedding_generated": False,
                "database_embeddings_checked": 0,
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "UNKNOWN",
                "threshold": MATCH_THRESHOLD,
                "final_decision": "ACCESS_DENIED",
                "decision_reason": "NO_FACE_DETECTED",
                "status": "denied",
                "person": "No Face Detected",
                "user_id": None,
                "similarity": 0.0,
                "annotated_image": annotated_base64,
                "message": "Access Denied: Position your face clearly in front of camera!"
            }

        # Check STEP 2: Both Eyes Detected Guard
        if not (left_eye_detected and right_eye_detected):
            log_access_event(db, "NO_VALID_EYES_DETECTED", "ACCESS_DENIED", 0.0)
            return {
                "face_detected": True,
                "left_eye_detected": left_eye_detected,
                "right_eye_detected": right_eye_detected,
                "liveness_passed": False,
                "query_embedding_generated": False,
                "database_embeddings_checked": 0,
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "UNKNOWN",
                "threshold": MATCH_THRESHOLD,
                "final_decision": "ACCESS_DENIED",
                "decision_reason": "NO_VALID_EYES_DETECTED",
                "status": "denied",
                "person": "No Face / Eye Detected",
                "user_id": None,
                "similarity": 0.0,
                "annotated_image": annotated_base64,
                "message": "Access Denied: Both eyes must be visible. Remove phone/hand from face!"
            }

        # STEP 3: Liveness Anti-Spoofing Check
        liveness_passed = check_spoof(polar_ribbon)
        if not liveness_passed:
            log_access_event(db, "UNKNOWN_SPOOF", "SPOOF_DETECTED", 0.0)
            return {
                "face_detected": True,
                "left_eye_detected": True,
                "right_eye_detected": True,
                "liveness_passed": False,
                "query_embedding_generated": False,
                "database_embeddings_checked": 0,
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "UNKNOWN",
                "threshold": MATCH_THRESHOLD,
                "final_decision": "ACCESS_DENIED",
                "decision_reason": "SPOOF_DETECTED",
                "status": "denied",
                "person": "SPOOF DETECTED (Fake Photo/Screen)",
                "user_id": None,
                "similarity": 0.0,
                "annotated_image": annotated_base64,
                "message": "Security Alert: Spoof attempt detected (photo printout or digital screen)."
            }

        # STEP 4: Generate 576-D L2-Normalized Query Iris Embedding
        tensor_img = transform(polar_ribbon).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = recognition_model.features(tensor_img)
            feat = recognition_model.avgpool(feat)
            feat = torch.flatten(feat, 1)
            query_feature = F.normalize(feat, p=2, dim=1)

        query_embedding_generated = True

        # STEP 5: Search ALL Enrolled Iris Templates in Database
        enrolled_embeddings = db.query(IrisEmbedding).all()
        database_embeddings_checked = len(enrolled_embeddings)

        if database_embeddings_checked == 0:
            log_access_event(db, "UNKNOWN", "ACCESS_DENIED", 0.0)
            return {
                "face_detected": True,
                "left_eye_detected": True,
                "right_eye_detected": True,
                "liveness_passed": True,
                "query_embedding_generated": True,
                "database_embeddings_checked": 0,
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "UNKNOWN",
                "threshold": MATCH_THRESHOLD,
                "final_decision": "ACCESS_DENIED",
                "decision_reason": "NO_REGISTERED_USERS_IN_DATABASE",
                "status": "denied",
                "person": "Unregistered / Unknown Person",
                "user_id": None,
                "similarity": 0.0,
                "annotated_image": annotated_base64,
                "message": "Access Denied: No users registered in database."
            }

        # STEP 6: Identify GLOBAL BEST MATCH across ALL database rows
        best_similarity = -1.0
        best_match_record = None

        for record in enrolled_embeddings:
            stored_vector = torch.tensor(record.embedding, device=device).unsqueeze(0)
            sim = F.cosine_similarity(query_feature, stored_vector, dim=1).item()

            if sim > best_similarity:
                best_similarity = sim
                best_match_record = record

        # Retrieve exact user identity associated with winning embedding record
        best_match_user = None
        if best_match_record is not None:
            best_match_user = db.query(User).filter(User.id == best_match_record.user_id).first()

        best_match_user_id = best_match_user.id if best_match_user else None
        best_match_user_name = best_match_user.name if best_match_user else "UNKNOWN"

        print(f"[GLOBAL MATCH SEARCH] Checked {database_embeddings_checked} DB Templates | Best Similarity: {best_similarity:.4f} | Winner: '{best_match_user_name}' (User #{best_match_user_id})")

        # STEP 7: Strict Match Threshold Evaluation
        if best_similarity >= MATCH_THRESHOLD:
            final_decision = "ACCESS_GRANTED"
            decision_reason = "MATCH_EXCEEDS_THRESHOLD"
            display_similarity = round(min(99.9, max(85.0, ((best_similarity - 0.82) / 0.18) * 14.5 + 85.0)), 2)
            person_display = best_match_user_name
            log_access_event(db, best_match_user_name, "ACCESS_GRANTED", best_similarity)
        else:
            final_decision = "ACCESS_DENIED"
            decision_reason = "NO_MATCH_ABOVE_THRESHOLD"
            display_similarity = round(max(0.0, (best_similarity / 0.82) * 35.0), 2)
            person_display = "Unregistered / Unknown Person"
            best_match_user_id = None
            log_access_event(db, "UNKNOWN", "ACCESS_DENIED", best_similarity)

        # STEP 8: Diagnostic Output Payload
        return {
            "face_detected": True,
            "left_eye_detected": True,
            "right_eye_detected": True,
            "liveness_passed": True,
            "query_embedding_generated": True,
            "database_embeddings_checked": database_embeddings_checked,
            "best_similarity": round(float(best_similarity), 4),
            "best_match_user_id": best_match_user_id if final_decision == "ACCESS_GRANTED" else None,
            "best_match_user_name": best_match_user_name if final_decision == "ACCESS_GRANTED" else "UNKNOWN",
            "threshold": MATCH_THRESHOLD,
            "final_decision": final_decision,
            "decision_reason": decision_reason,
            "status": "success" if final_decision == "ACCESS_GRANTED" else "denied",
            "person": person_display,
            "user_id": best_match_user_id if final_decision == "ACCESS_GRANTED" else None,
            "similarity": display_similarity,
            "annotated_image": annotated_base64,
            "message": f"Welcome, {best_match_user_name}!" if final_decision == "ACCESS_GRANTED" else "Access Denied: Iris pattern does not match any registered user."
        }

    except Exception as e:
        print("[Recognition Exception]:", e)
        return {
            "face_detected": False,
            "left_eye_detected": False,
            "right_eye_detected": False,
            "liveness_passed": False,
            "query_embedding_generated": False,
            "database_embeddings_checked": 0,
            "best_similarity": 0.0,
            "best_match_user_id": None,
            "best_match_user_name": "UNKNOWN",
            "threshold": MATCH_THRESHOLD,
            "final_decision": "ACCESS_DENIED",
            "decision_reason": f"SYSTEM_ERROR: {str(e)}",
            "status": "error",
            "person": "System Error",
            "similarity": 0.0,
            "message": str(e)
        }
    finally:
        db.close()


def recognize_iris_v2_diagnostic(image_input):
    """
    Phase 5C — Final V2 Verification Pipeline in Diagnostic Mode.
    
    Features:
    1. Phase 3 Landmark Eye & Iris Localization.
    2. Quality Gate Verification.
    3. Shared V2 Preprocessing (64x256 RGB ImageNet Normalized Tensor).
    4. Eye-Side Isolated Global Template Matching (L->L, R->R).
    5. Dual-Eye Identity Consistency & Conflict Check.
    6. Diagnostic Threshold Evaluation (DIAGNOSTIC_THRESHOLD = 0.3288).
    7. Hardware Action Bypassed (hardware_action = "NONE").
    """
    db: Session = SessionLocal()
    DIAGNOSTIC_THRESHOLD = 0.3288

    try:
        import numpy as np
        import cv2
        from app.models.user_model import IrisEmbeddingV2
        from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            img_np = np.array(pil_image)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input.copy()
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(img_np)
        else:
            img_np = cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_np)

        res = detect_and_crop_iris(img_np)
        payload = res.payload

        # Base64 debug annotated image
        buffered = io.BytesIO()
        res[1].save(buffered, format="JPEG")
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        # Check 1: Face Detection Guard
        if not payload.get("face_detected"):
            return {
                "face_detected": False,
                "left_eye_detected": False,
                "right_eye_detected": False,
                "left_embedding_generated": False,
                "right_embedding_generated": False,
                "database_templates_checked": 0,
                "left_templates_checked": 0,
                "right_templates_checked": 0,
                "left_best_similarity": 0.0,
                "right_best_similarity": 0.0,
                "left_best_user_id": None,
                "right_best_user_id": None,
                "left_best_user_name": "NONE",
                "right_best_user_name": "NONE",
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "Unregistered / Unknown Person",
                "winning_user_id": None,
                "winning_user_name": "Unregistered / Unknown Person",
                "winning_user_score": 0.0,
                "second_best_user_id": None,
                "second_best_user_score": 0.0,
                "candidate_margin": 0.0,
                "user_candidate_scores": [],
                "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
                "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
                "decision_reason": "NO_FACE_DETECTED",
                "hardware_action": "NONE",
                "annotated_image": annotated_base64,
                "message": "V2 Diagnostic: Position face clearly in front of camera!"
            }

        left_iris_det = payload.get("left_iris_detected", False)
        right_iris_det = payload.get("right_iris_detected", False)

        # Check 2: Quality Gate Guard
        if not (left_iris_det or right_iris_det):
            return {
                "face_detected": True,
                "left_eye_detected": payload.get("left_eye_detected", False),
                "right_eye_detected": payload.get("right_eye_detected", False),
                "left_embedding_generated": False,
                "right_embedding_generated": False,
                "database_templates_checked": 0,
                "left_templates_checked": 0,
                "right_templates_checked": 0,
                "left_best_similarity": 0.0,
                "right_best_similarity": 0.0,
                "left_best_user_id": None,
                "right_best_user_id": None,
                "left_best_user_name": "NONE",
                "right_best_user_name": "NONE",
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "Unregistered / Unknown Person",
                "winning_user_id": None,
                "winning_user_name": "Unregistered / Unknown Person",
                "winning_user_score": 0.0,
                "second_best_user_id": None,
                "second_best_user_score": 0.0,
                "candidate_margin": 0.0,
                "user_candidate_scores": [],
                "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
                "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
                "decision_reason": payload.get("failure_reason", "QUALITY_GATE_FAILED"),
                "hardware_action": "NONE",
                "annotated_image": annotated_base64,
                "message": f"V2 Diagnostic: Quality gate failed ({payload.get('failure_reason')})."
            }

        # Group enrolled templates by User and compute User-Level Aggregated Evidence
        enrolled_users = db.query(User).all()
        left_v2_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.eye_side == "L").all()
        right_v2_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.eye_side == "R").all()

        left_count = len(left_v2_recs)
        right_count = len(right_v2_recs)
        total_checked = left_count + right_count

        user_candidate_scores = []
        left_emb_gen = False
        right_emb_gen = False

        # Extract Left Query Embedding
        left_tensor = None
        if left_iris_det and left_count > 0:
            lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
            left_crop = Image.fromarray(img_np[ly1:ly2, lx1:lx2])
            left_vec = extract_v2_embedding(left_crop)
            left_tensor = torch.tensor(left_vec, device=device).unsqueeze(0)
            left_emb_gen = True

        # Extract Right Query Embedding
        right_tensor = None
        if right_iris_det and right_count > 0:
            rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
            right_crop = Image.fromarray(img_np[ry1:ry2, rx1:rx2])
            right_vec = extract_v2_embedding(right_crop)
            right_tensor = torch.tensor(right_vec, device=device).unsqueeze(0)
            right_emb_gen = True

        # User-Level Multi-Template Aggregation Calculation
        for u in enrolled_users:
            u_left_recs = [r for r in left_v2_recs if r.user_id == u.id]
            u_right_recs = [r for r in right_v2_recs if r.user_id == u.id]

            if not u_left_recs and not u_right_recs:
                continue

            l_sims = []
            l_best = -1.0
            if left_tensor is not None and u_left_recs:
                for rec in u_left_recs:
                    st_t = torch.tensor(rec.embedding, device=device).unsqueeze(0)
                    sim = F.cosine_similarity(left_tensor, st_t, dim=1).item()
                    l_sims.append(sim)
                    if sim > l_best:
                        l_best = sim

            r_sims = []
            r_best = -1.0
            if right_tensor is not None and u_right_recs:
                for rec in u_right_recs:
                    st_t = torch.tensor(rec.embedding, device=device).unsqueeze(0)
                    sim = F.cosine_similarity(right_tensor, st_t, dim=1).item()
                    r_sims.append(sim)
                    if sim > r_best:
                        r_best = sim

            # Top-2 aggregated mean (robust against single template outliers)
            l_top_k = sorted(l_sims, reverse=True)[:min(2, len(l_sims))] if l_sims else []
            r_top_k = sorted(r_sims, reverse=True)[:min(2, len(r_sims))] if r_sims else []

            l_agg = float(np.mean(l_top_k)) if l_top_k else 0.0
            r_agg = float(np.mean(r_top_k)) if r_top_k else 0.0

            l_best_val = float(l_best) if l_sims else 0.0
            r_best_val = float(r_best) if r_sims else 0.0

            # Combined User Score calculation
            if l_sims and r_sims:
                combined_score = (l_agg + r_agg) / 2.0
            elif l_sims:
                combined_score = l_agg
            elif r_sims:
                combined_score = r_agg
            else:
                combined_score = 0.0

            user_candidate_scores.append({
                "user_id": u.id,
                "user_name": u.name,
                "left_template_count": len(u_left_recs),
                "right_template_count": len(u_right_recs),
                "left_best_similarity": round(l_best_val, 4),
                "left_aggregated_similarity": round(l_agg, 4),
                "right_best_similarity": round(r_best_val, 4),
                "right_aggregated_similarity": round(r_agg, 4),
                "combined_user_score": round(combined_score, 4)
            })

        # Sort User Candidates by Combined User Score Descending
        user_candidate_scores.sort(key=lambda x: x["combined_user_score"], reverse=True)

        if not user_candidate_scores:
            return {
                "face_detected": True,
                "left_eye_detected": payload.get("left_eye_detected", False),
                "right_eye_detected": payload.get("right_eye_detected", False),
                "left_embedding_generated": left_emb_gen,
                "right_embedding_generated": right_emb_gen,
                "database_templates_checked": 0,
                "left_templates_checked": 0,
                "right_templates_checked": 0,
                "left_best_similarity": 0.0,
                "right_best_similarity": 0.0,
                "left_best_user_id": None,
                "right_best_user_id": None,
                "left_best_user_name": "NONE",
                "right_best_user_name": "NONE",
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "Unregistered / Unknown Person",
                "best_matching_template_id": None,
                "winning_user_id": None,
                "winning_user_name": "Unregistered / Unknown Person",
                "winning_user_score": 0.0,
                "second_best_user_id": None,
                "second_best_user_score": 0.0,
                "candidate_margin": 0.0,
                "user_candidate_scores": [],
                "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
                "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
                "decision_reason": "NO_V2_ENROLLED_USERS",
                "hardware_action": "NONE",
                "annotated_image": annotated_base64,
                "message": "V2 Diagnostic: No enrolled V2 users found in database."
            }

        # Rank Top Candidates
        winner = user_candidate_scores[0]
        second = user_candidate_scores[1] if len(user_candidate_scores) > 1 else None

        winning_user_id = winner["user_id"]
        winning_user_name = winner["user_name"]
        winning_user_score = winner["combined_user_score"]

        second_user_id = second["user_id"] if second else None
        second_user_score = second["combined_user_score"] if second else 0.0
        candidate_margin = round(winning_user_score - second_user_score, 4)

        # Eye-Specific User Winners (based on aggregated eye similarities)
        l_eye_winner = max(user_candidate_scores, key=lambda x: x["left_aggregated_similarity"]) if left_emb_gen else None
        r_eye_winner = max(user_candidate_scores, key=lambda x: x["right_aggregated_similarity"]) if right_emb_gen else None

        left_best_user_id = l_eye_winner["user_id"] if (l_eye_winner and l_eye_winner["left_aggregated_similarity"] >= DIAGNOSTIC_THRESHOLD) else None
        left_best_user_name = l_eye_winner["user_name"] if left_best_user_id else "NONE"
        left_best_sim = l_eye_winner["left_aggregated_similarity"] if l_eye_winner else 0.0

        right_best_user_id = r_eye_winner["user_id"] if (r_eye_winner and r_eye_winner["right_aggregated_similarity"] >= DIAGNOSTIC_THRESHOLD) else None
        right_best_user_name = r_eye_winner["user_name"] if right_best_user_id else "NONE"
        right_best_sim = r_eye_winner["right_aggregated_similarity"] if r_eye_winner else 0.0

        left_valid = (left_best_user_id is not None)
        right_valid = (right_best_user_id is not None)

        # Dual-Eye Consistency & Diagnostic Decision Logic with Candidate Margin Ambiguity Guard
        MIN_CANDIDATE_MARGIN = 0.08  # Enforce at least 8% score margin over 2nd best candidate

        if winning_user_score >= DIAGNOSTIC_THRESHOLD:
            if left_valid and right_valid:
                if left_best_user_id == right_best_user_id:
                    # Enforce candidate separation margin guard
                    if second is not None and candidate_margin < MIN_CANDIDATE_MARGIN:
                        final_status = "V2_DIAGNOSTIC_UNMATCHED"
                        final_winner_id = None
                        final_winner_name = "Unregistered / Unknown Person"
                        reason = f"AMBIGUOUS_RGB_CANDIDATE_COLLISION: Candidate #{winning_user_id} ({winning_user_name}, score={winning_user_score:.4f}) margin over 2nd candidate #{second_user_id} ({second['user_name']}, score={second_user_score:.4f}) is {candidate_margin:.4f} < {MIN_CANDIDATE_MARGIN:.2f}"
                    else:
                        final_status = "V2_DIAGNOSTIC_MATCH"
                        final_winner_id = winning_user_id
                        final_winner_name = winning_user_name
                        reason = f"DUAL_EYE_CONSISTENT_MATCH (User #{winning_user_id} {winning_user_name}, Score={winning_user_score:.4f}, Margin={candidate_margin:.4f})"
                else:
                    final_status = "V2_DIAGNOSTIC_CONFLICT"
                    final_winner_id = None
                    final_winner_name = "IDENTITY_CONFLICT"
                    reason = f"IDENTITY_CONFLICT: Left eye prefers {left_best_user_name} (ID #{left_best_user_id}) but Right eye prefers {right_best_user_name} (ID #{right_best_user_id})"
            elif left_valid:
                if second is not None and candidate_margin < MIN_CANDIDATE_MARGIN:
                    final_status = "V2_DIAGNOSTIC_UNMATCHED"
                    final_winner_id = None
                    final_winner_name = "Unregistered / Unknown Person"
                    reason = f"AMBIGUOUS_SINGLE_EYE_MARGIN: Margin {candidate_margin:.4f} < {MIN_CANDIDATE_MARGIN:.2f}"
                else:
                    final_status = "V2_DIAGNOSTIC_MATCH"
                    final_winner_id = left_best_user_id
                    final_winner_name = left_best_user_name
                    reason = f"SINGLE_EYE_LEFT_MATCH (User #{left_best_user_id} {left_best_user_name})"
            elif right_valid:
                if second is not None and candidate_margin < MIN_CANDIDATE_MARGIN:
                    final_status = "V2_DIAGNOSTIC_UNMATCHED"
                    final_winner_id = None
                    final_winner_name = "Unregistered / Unknown Person"
                    reason = f"AMBIGUOUS_SINGLE_EYE_MARGIN: Margin {candidate_margin:.4f} < {MIN_CANDIDATE_MARGIN:.2f}"
                else:
                    final_status = "V2_DIAGNOSTIC_MATCH"
                    final_winner_id = right_best_user_id
                    final_winner_name = right_best_user_name
                    reason = f"SINGLE_EYE_RIGHT_MATCH (User #{right_best_user_id} {right_best_user_name})"
            else:
                final_status = "V2_DIAGNOSTIC_UNMATCHED"
                final_winner_id = None
                final_winner_name = "Unregistered / Unknown Person"
                reason = f"AGGREGATED_SCORE_BELOW_DIAGNOSTIC_THRESHOLD (Score={winning_user_score:.4f} < {DIAGNOSTIC_THRESHOLD})"
        else:
            final_status = "V2_DIAGNOSTIC_UNMATCHED"
            final_winner_id = None
            final_winner_name = "Unregistered / Unknown Person"
            reason = f"AGGREGATED_SCORE_BELOW_DIAGNOSTIC_THRESHOLD (Top Score={winning_user_score:.4f} < {DIAGNOSTIC_THRESHOLD})"

        print(f"[V2 DIAGNOSTIC VERIFICATION] Status: {final_status} | Winner: '{final_winner_name}' (ID #{final_winner_id}) | Score: {winning_user_score:.4f} | Margin: {candidate_margin:.4f}")

        return {
            "face_detected": True,
            "left_eye_detected": payload.get("left_eye_detected", False),
            "right_eye_detected": payload.get("right_eye_detected", False),
            "left_embedding_generated": left_emb_gen,
            "right_embedding_generated": right_emb_gen,
            "database_templates_checked": total_checked,
            "left_templates_checked": left_count,
            "right_templates_checked": right_count,
            "left_best_similarity": left_best_sim,
            "right_best_similarity": right_best_sim,
            "left_best_user_id": left_best_user_id,
            "right_best_user_id": right_best_user_id,
            "left_best_user_name": left_best_user_name,
            "right_best_user_name": right_best_user_name,
            "best_similarity": winning_user_score,
            "best_match_user_id": final_winner_id,
            "best_match_user_name": final_winner_name,
            "best_matching_template_id": None,
            "winning_user_id": final_winner_id,
            "winning_user_name": final_winner_name,
            "winning_user_score": winning_user_score,
            "second_best_user_id": second_user_id,
            "second_best_user_score": second_user_score,
            "candidate_margin": candidate_margin,
            "user_candidate_scores": user_candidate_scores,
            "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
            "final_diagnostic_status": final_status,
            "decision_reason": reason,
            "hardware_action": "NONE",
            "annotated_image": annotated_base64,
            "message": f"V2 Diagnostic Result: {final_status} ({reason})"
        }

    except Exception as e:
        print("[V2 Recognition Diagnostic Exception]:", e)
        return {
            "face_detected": False,
            "left_eye_detected": False,
            "right_eye_detected": False,
            "left_embedding_generated": False,
            "right_embedding_generated": False,
            "database_templates_checked": 0,
            "left_templates_checked": 0,
            "right_templates_checked": 0,
            "left_best_similarity": 0.0,
            "right_best_similarity": 0.0,
            "left_best_user_id": None,
            "right_best_user_id": None,
            "left_best_user_name": "NONE",
            "right_best_user_name": "NONE",
            "best_similarity": 0.0,
            "best_match_user_id": None,
            "best_match_user_name": "Unregistered / Unknown Person",
            "best_matching_template_id": None,
            "winning_user_id": None,
            "winning_user_name": "Unregistered / Unknown Person",
            "winning_user_score": 0.0,
            "second_best_user_id": None,
            "second_best_user_score": 0.0,
            "candidate_margin": 0.0,
            "user_candidate_scores": [],
            "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
            "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
            "decision_reason": f"ERROR: {str(e)}",
            "hardware_action": "NONE",
            "message": str(e)
        }
    finally:
        db.close()


def recognize_iris_v2_sequence_diagnostic(images_input, min_margin: float = 0.08):
    """
    Multi-Frame Sequential Capture & Multi-Frame Temporal Evidence Aggregation Voting Engine
    Workflow:
    Laptop Camera -> Capture 5-7 frames -> Detect both eyes -> Generate embeddings per frame ->
    Compare L->L and R->R -> Aggregate evidence PER USER -> Temporal voting -> Require SAME USER consistently ->
    Clear winner? YES -> MATCH | NO -> UNMATCHED
    """
    db = SessionLocal()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DIAGNOSTIC_THRESHOLD = 0.3288

    try:
        enrolled_users = db.query(User).all()
        left_v2_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.eye_side == "L").all()
        right_v2_recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.eye_side == "R").all()

        frame_results = []
        user_frame_votes = {u.id: 0 for u in enrolled_users}

        user_left_sims = {u.id: [] for u in enrolled_users}
        user_right_sims = {u.id: [] for u in enrolled_users}

        last_annotated_base64 = None

        for idx, img_src in enumerate(images_input):
            if isinstance(img_src, str):
                img_np = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_np)
            elif isinstance(img_src, Image.Image):
                pil_img = img_src.convert("RGB")
                img_np = np.array(pil_img)
            else:
                img_np = np.array(img_src)
                pil_img = Image.fromarray(img_np)

            res = detect_and_crop_iris(img_np)
            payload = res.payload

            # Base64 debug annotated image
            buffered = io.BytesIO()
            res[1].save(buffered, format="JPEG")
            last_annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

            if not payload.get("face_detected"):
                frame_results.append({"frame_index": idx, "status": "NO_FACE"})
                continue

            left_iris_det = payload.get("left_iris_detected", False)
            right_iris_det = payload.get("right_iris_detected", False)

            if not (left_iris_det or right_iris_det):
                frame_results.append({"frame_index": idx, "status": "QUALITY_FAILED"})
                continue

            left_tensor = None
            if left_iris_det and payload.get("left_eye_bbox"):
                lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
                left_crop = Image.fromarray(img_np[ly1:ly2, lx1:lx2])
                left_vec = extract_v2_embedding(left_crop)
                left_tensor = torch.tensor(left_vec, device=device).unsqueeze(0)

            right_tensor = None
            if right_iris_det and payload.get("right_eye_bbox"):
                rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
                right_crop = Image.fromarray(img_np[ry1:ry2, rx1:rx2])
                right_vec = extract_v2_embedding(right_crop)
                right_tensor = torch.tensor(right_vec, device=device).unsqueeze(0)

            frame_candidate_scores = []

            for u in enrolled_users:
                u_l_recs = [r for r in left_v2_recs if r.user_id == u.id]
                u_r_recs = [r for r in right_v2_recs if r.user_id == u.id]

                if not u_l_recs and not u_r_recs:
                    continue

                l_sims = []
                if left_tensor is not None and u_l_recs:
                    for rec in u_l_recs:
                        st_t = torch.tensor(rec.embedding, device=device).unsqueeze(0)
                        sim = float(F.cosine_similarity(left_tensor, st_t, dim=1).item())
                        l_sims.append(sim)
                        user_left_sims[u.id].append(sim)

                r_sims = []
                if right_tensor is not None and u_r_recs:
                    for rec in u_r_recs:
                        st_t = torch.tensor(rec.embedding, device=device).unsqueeze(0)
                        sim = float(F.cosine_similarity(right_tensor, st_t, dim=1).item())
                        r_sims.append(sim)
                        user_right_sims[u.id].append(sim)

                l_agg = float(np.mean(sorted(l_sims, reverse=True)[:2])) if l_sims else 0.0
                r_agg = float(np.mean(sorted(r_sims, reverse=True)[:2])) if r_sims else 0.0

                comb = (l_agg + r_agg) / 2.0 if (l_sims and r_sims) else (l_agg or r_agg)
                frame_candidate_scores.append({"user_id": u.id, "user_name": u.name, "score": comb})

            frame_candidate_scores.sort(key=lambda x: x["score"], reverse=True)
            if frame_candidate_scores:
                frame_winner = frame_candidate_scores[0]
                user_frame_votes[frame_winner["user_id"]] += 1
                frame_results.append({
                    "frame_index": idx,
                    "status": "VALID",
                    "winner_id": frame_winner["user_id"],
                    "winner_name": frame_winner["user_name"],
                    "winner_score": round(frame_winner["score"], 4)
                })

        # Calculate Multi-Frame Per-User Evidence Aggregation
        user_sequence_scores = []
        total_valid_frames = len([f for f in frame_results if f.get("status") == "VALID"])

        for u in enrolled_users:
            l_sims = user_left_sims[u.id]
            r_sims = user_right_sims[u.id]

            if not l_sims and not r_sims:
                continue

            l_top3 = sorted(l_sims, reverse=True)[:min(3, len(l_sims))] if l_sims else []
            r_top3 = sorted(r_sims, reverse=True)[:min(3, len(r_sims))] if r_sims else []

            l_seq_agg = float(np.mean(l_top3)) if l_top3 else 0.0
            r_seq_agg = float(np.mean(r_top3)) if r_top3 else 0.0

            seq_comb = (l_seq_agg + r_seq_agg) / 2.0 if (l_sims and r_sims) else (l_seq_agg or r_seq_agg)
            votes = user_frame_votes[u.id]
            vote_pct = (votes / total_valid_frames * 100.0) if total_valid_frames > 0 else 0.0

            user_sequence_scores.append({
                "user_id": u.id,
                "user_name": u.name,
                "frame_votes": votes,
                "vote_percentage": round(vote_pct, 1),
                "left_seq_aggregated": round(l_seq_agg, 4),
                "right_seq_aggregated": round(r_seq_agg, 4),
                "combined_sequence_score": round(seq_comb, 4)
            })

        user_sequence_scores.sort(key=lambda x: x["combined_sequence_score"], reverse=True)

        if not user_sequence_scores or total_valid_frames == 0:
            return {
                "face_detected": False,
                "left_eye_detected": False,
                "right_eye_detected": False,
                "left_embedding_generated": False,
                "right_embedding_generated": False,
                "database_templates_checked": len(left_v2_recs) + len(right_v2_recs),
                "left_templates_checked": len(left_v2_recs),
                "right_templates_checked": len(right_v2_recs),
                "left_best_similarity": 0.0,
                "right_best_similarity": 0.0,
                "left_best_user_id": None,
                "right_best_user_id": None,
                "left_best_user_name": "NONE",
                "right_best_user_name": "NONE",
                "best_similarity": 0.0,
                "best_match_user_id": None,
                "best_match_user_name": "Unregistered / Unknown Person",
                "winning_user_id": None,
                "winning_user_name": "Unregistered / Unknown Person",
                "winning_user_score": 0.0,
                "second_best_user_id": None,
                "second_best_user_score": 0.0,
                "candidate_margin": 0.0,
                "user_candidate_scores": [],
                "temporal_votes": user_frame_votes,
                "total_frames_processed": len(images_input),
                "valid_frames_count": 0,
                "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
                "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
                "decision_reason": "NO_VALID_FRAMES_IN_SEQUENCE",
                "hardware_action": "NONE",
                "annotated_image": last_annotated_base64,
                "message": "V2 Diagnostic Sequence: No valid eye frames detected in sequence."
            }

        seq_winner = user_sequence_scores[0]
        seq_second = user_sequence_scores[1] if len(user_sequence_scores) > 1 else None

        winner_id = seq_winner["user_id"]
        winner_name = seq_winner["user_name"]
        winner_score = seq_winner["combined_sequence_score"]
        second_user_id = seq_second["user_id"] if seq_second else None
        second_score = seq_second["combined_sequence_score"] if seq_second else 0.0
        candidate_margin = round(winner_score - second_score, 4)

        # Majority Voting Rule (>= 60% of valid frames must vote for SAME USER)
        vote_passed = (seq_winner["vote_percentage"] >= 60.0)
        margin_passed = (candidate_margin >= min_margin)
        cutoff_passed = (winner_score >= DIAGNOSTIC_THRESHOLD)

        if cutoff_passed and vote_passed and margin_passed:
            final_status = "V2_DIAGNOSTIC_MATCH"
            final_winner_id = winner_id
            final_winner_name = winner_name
            reason = f"TEMPORAL_MAJORITY_MATCH (User #{winner_id} {winner_name}, Votes={seq_winner['frame_votes']}/{total_valid_frames} [{seq_winner['vote_percentage']}%], Score={winner_score:.4f}, Margin={candidate_margin:.4f})"
        elif cutoff_passed and not vote_passed:
            final_status = "V2_DIAGNOSTIC_CONFLICT"
            final_winner_id = None
            final_winner_name = "IDENTITY_CONFLICT"
            reason = f"TEMPORAL_VOTING_CONFLICT: Top candidate received only {seq_winner['vote_percentage']}% of frame votes (< 60.0%)"
        elif cutoff_passed and not margin_passed:
            final_status = "V2_DIAGNOSTIC_UNMATCHED"
            final_winner_id = None
            final_winner_name = "Unregistered / Unknown Person"
            reason = f"AMBIGUOUS_SEQUENCE_MARGIN: Candidate margin {candidate_margin:.4f} < {min_margin:.2f}"
        else:
            final_status = "V2_DIAGNOSTIC_UNMATCHED"
            final_winner_id = None
            final_winner_name = "Unregistered / Unknown Person"
            reason = f"SEQUENCE_SCORE_BELOW_DIAGNOSTIC_THRESHOLD ({winner_score:.4f} < {DIAGNOSTIC_THRESHOLD})"

        print(f"[V2 SEQUENCE VERIFICATION] Status: {final_status} | Winner: '{final_winner_name}' (ID #{final_winner_id}) | Score: {winner_score:.4f} | Margin: {candidate_margin:.4f} | Reason: {reason}")

        return {
            "face_detected": True,
            "left_eye_detected": True,
            "right_eye_detected": True,
            "left_embedding_generated": True,
            "right_embedding_generated": True,
            "database_templates_checked": len(left_v2_recs) + len(right_v2_recs),
            "left_templates_checked": len(left_v2_recs),
            "right_templates_checked": len(right_v2_recs),
            "left_best_similarity": seq_winner["left_seq_aggregated"],
            "right_best_similarity": seq_winner["right_seq_aggregated"],
            "left_best_user_id": final_winner_id,
            "right_best_user_id": final_winner_id,
            "left_best_user_name": final_winner_name,
            "right_best_user_name": final_winner_name,
            "best_similarity": winner_score,
            "best_match_user_id": final_winner_id,
            "best_match_user_name": final_winner_name,
            "best_matching_template_id": None,
            "winning_user_id": final_winner_id,
            "winning_user_name": final_winner_name,
            "winning_user_score": winner_score,
            "second_best_user_id": second_user_id,
            "second_best_user_score": second_score,
            "candidate_margin": candidate_margin,
            "user_candidate_scores": user_sequence_scores,
            "temporal_votes": user_frame_votes,
            "total_frames_processed": len(images_input),
            "valid_frames_count": total_valid_frames,
            "frame_results": frame_results,
            "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
            "final_diagnostic_status": final_status,
            "decision_reason": reason,
            "hardware_action": "NONE",
            "annotated_image": last_annotated_base64,
            "message": f"V2 Diagnostic Sequence Result: {final_status} ({reason})"
        }

    except Exception as e:
        print("[V2 Sequence Recognition Diagnostic Exception]:", e)
        return {
            "face_detected": False,
            "left_eye_detected": False,
            "right_eye_detected": False,
            "left_embedding_generated": False,
            "right_embedding_generated": False,
            "database_templates_checked": 0,
            "left_templates_checked": 0,
            "right_templates_checked": 0,
            "left_best_similarity": 0.0,
            "right_best_similarity": 0.0,
            "left_best_user_id": None,
            "right_best_user_id": None,
            "left_best_user_name": "NONE",
            "right_best_user_name": "NONE",
            "best_similarity": 0.0,
            "best_match_user_id": None,
            "best_match_user_name": "Unregistered / Unknown Person",
            "winning_user_id": None,
            "winning_user_name": "Unregistered / Unknown Person",
            "winning_user_score": 0.0,
            "second_best_user_id": None,
            "second_best_user_score": 0.0,
            "candidate_margin": 0.0,
            "user_candidate_scores": [],
            "temporal_votes": {},
            "total_frames_processed": len(images_input),
            "valid_frames_count": 0,
            "frame_results": [],
            "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
            "final_diagnostic_status": "V2_DIAGNOSTIC_UNMATCHED",
            "decision_reason": f"ERROR: {str(e)}",
            "hardware_action": "NONE",
            "message": str(e)
        }
    finally:
        db.close()


def recognize_iris_v2_1to1_diagnostic(
    image_input,
    claimed_user_id: int,
    action: str = "CHECK-IN",
    candidate_only: bool = False,
    dry_run: bool = False
):
    """
    Temporary Laptop RGB Webcam Demo Mode — 1:1 Claimed Identity Verification Engine.
    When candidate_only=True: compares live iris ONLY against claimed_user_id's enrolled templates
    (no global 1:N iris search, candidate fixes verification target).
    When dry_run=True: skips access_logs logging and suppresses hardware triggers.
    """
    db = SessionLocal()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RGB_1TO1_THRESHOLD = DEFAULT_IRIS_MATCH_THRESHOLD
    MIN_IDENTITY_MARGIN = DEFAULT_IRIS_MIN_IDENTITY_MARGIN
    is_dry = bool(dry_run) or (str(action).upper() in ["DIAGNOSTIC", "DRY-RUN", "DRY_RUN", "TEST"])

    try:
        claimed_user = db.query(User).filter(User.id == claimed_user_id).first()
        if not claimed_user:
            return {
                "face_detected": False,
                "claimed_user_id": claimed_user_id,
                "claimed_user_name": f"User ID #{claimed_user_id} Not Found",
                "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
                "decision_reason": f"CLAIMED_USER_NOT_FOUND: User ID #{claimed_user_id} does not exist",
                "biometric_verified": False,
                "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
                "template_ids_used": [],
                "templates_checked": 0
            }

        # Process input image
        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
            img_np = np.array(pil_image)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
            pil_image = Image.fromarray(img_np)
        elif isinstance(image_input, (bytes, bytearray)):
            pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
            img_np = np.array(pil_image)
        else:
            img_np = cv2.cvtColor(cv2.imread(image_input), cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(img_np)

        res = detect_and_crop_iris(img_np)
        payload = res.payload

        buffered = io.BytesIO()
        res[1].save(buffered, format="JPEG")
        annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        if not payload.get("face_detected"):
            if not is_dry:
                log_access_event(db, claimed_user.name, "ACCESS_DENIED", 0.0, action=action)
            return {
                "face_detected": False,
                "left_eye_detected": False,
                "right_eye_detected": False,
                "claimed_user_id": claimed_user.id,
                "claimed_user_name": claimed_user.name,
                "best_similarity": 0.0,
                "strongest_competing_score": 0.0,
                "identity_margin": 0.0,
                "left_eye_consistency": "FAIL",
                "right_eye_consistency": "FAIL",
                "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
                "decision_reason": "NO_FACE_DETECTED",
                "biometric_verified": False,
                "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
                "template_ids_used": [],
                "templates_checked": 0,
                "annotated_image": annotated_base64,
                "message": "1:1 Verification: Position face clearly in front of camera!"
            }

        left_iris_det = payload.get("left_iris_detected", False)
        right_iris_det = payload.get("right_iris_detected", False)

        if not (left_iris_det or right_iris_det):
            if not is_dry:
                log_access_event(db, claimed_user.name, "ACCESS_DENIED", 0.0, action=action)
            return {
                "face_detected": True,
                "left_eye_detected": payload.get("left_eye_detected", False),
                "right_eye_detected": payload.get("right_eye_detected", False),
                "claimed_user_id": claimed_user.id,
                "claimed_user_name": claimed_user.name,
                "best_similarity": 0.0,
                "strongest_competing_score": 0.0,
                "identity_margin": 0.0,
                "left_eye_consistency": "FAIL",
                "right_eye_consistency": "FAIL",
                "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
                "decision_reason": "QUALITY_FAILED",
                "biometric_verified": False,
                "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
                "template_ids_used": [],
                "templates_checked": 0,
                "annotated_image": annotated_base64,
                "message": "1:1 Verification: Iris ROI quality check failed."
            }

        # Query templates: candidate-only vs global
        if candidate_only:
            recs = db.query(IrisEmbeddingV2).filter(IrisEmbeddingV2.user_id == claimed_user.id).all()
            template_ids_used = [r.id for r in recs]
            user_templates = {claimed_user.id: {"L": [], "R": []}}
            for r in recs:
                user_templates[claimed_user.id][r.eye_side].append(r.embedding)
        else:
            recs = db.query(IrisEmbeddingV2).all()
            template_ids_used = [r.id for r in recs]
            user_templates = {}
            for r in recs:
                if r.user_id not in user_templates:
                    user_templates[r.user_id] = {"L": [], "R": []}
                user_templates[r.user_id][r.eye_side].append(r.embedding)

        # Extract embeddings
        left_vec = None
        if left_iris_det and payload.get("left_eye_bbox"):
            lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
            left_crop = Image.fromarray(img_np[ly1:ly2, lx1:lx2])
            left_vec = extract_v2_embedding(left_crop)

        right_vec = None
        if right_iris_det and payload.get("right_eye_bbox"):
            rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
            right_crop = Image.fromarray(img_np[ry1:ry2, rx1:rx2])
            right_vec = extract_v2_embedding(right_crop)

        # Calculate scores for candidate templates
        user_scores = {}
        left_winner_id = None
        left_winner_score = -1.0
        right_winner_id = None
        right_winner_score = -1.0

        for uid, t_dict in user_templates.items():
            l_sims = []
            if left_vec and t_dict["L"]:
                lt = torch.tensor(left_vec, device=device).unsqueeze(0)
                for emb in t_dict["L"]:
                    st = torch.tensor(emb, device=device).unsqueeze(0)
                    sim = float(F.cosine_similarity(lt, st, dim=1).item())
                    l_sims.append(sim)

            r_sims = []
            if right_vec and t_dict["R"]:
                rt = torch.tensor(right_vec, device=device).unsqueeze(0)
                for emb in t_dict["R"]:
                    st = torch.tensor(emb, device=device).unsqueeze(0)
                    sim = float(F.cosine_similarity(rt, st, dim=1).item())
                    r_sims.append(sim)

            l_agg = float(np.mean(sorted(l_sims, reverse=True)[:2])) if l_sims else 0.0
            r_agg = float(np.mean(sorted(r_sims, reverse=True)[:2])) if r_sims else 0.0

            if l_agg > left_winner_score:
                left_winner_score = l_agg
                left_winner_id = uid
            if r_agg > right_winner_score:
                right_winner_score = r_agg
                right_winner_id = uid

            u_score = (l_agg + r_agg) / 2.0 if (l_sims and r_sims) else (l_agg or r_agg)
            user_scores[uid] = u_score

        claimed_score = user_scores.get(claimed_user.id, 0.0)

        other_scores = [sc for uid, sc in user_scores.items() if uid != claimed_user.id]
        strongest_other_score = float(max(other_scores)) if other_scores else 0.0
        identity_margin = claimed_score - strongest_other_score

        left_consistency = "PASS" if (left_winner_id == claimed_user.id) else "FAIL"
        right_consistency = "PASS" if (right_winner_id == claimed_user.id) else "FAIL"

        # Biometric Decision Rules
        biometrics_passed = True
        if claimed_score < RGB_1TO1_THRESHOLD:
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "SCORE_BELOW_THRESHOLD"
            biometrics_passed = False
        elif not candidate_only and identity_margin < MIN_IDENTITY_MARGIN:
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "COMPETING_IDENTITY_TOO_CLOSE"
            biometrics_passed = False
        elif not candidate_only and ((left_vec and left_consistency == "FAIL") or (right_vec and right_consistency == "FAIL")):
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "EYE_IDENTITY_CONFLICT"
            biometrics_passed = False

        check_in_time_iso = None
        check_in_time_fmt = None
        check_out_time_iso = None
        check_out_time_fmt = None
        total_duration_str = None

        if not biometrics_passed:
            db_status = "ACCESS_DENIED"
            saved_log = None
            if not is_dry:
                saved_log = log_access_event(db, claimed_user.name, db_status, claimed_score, action=action)
        else:
            is_inside, last_in_log = is_user_currently_inside(db, claimed_user.name)

            if action == "CHECK-IN":
                if is_inside:
                    final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
                    decision_reason = "USER_ALREADY_INSIDE"
                    db_status = "ACCESS_DENIED"
                    saved_log = None
                    if last_in_log and last_in_log.timestamp:
                        check_in_time_iso = last_in_log.timestamp.isoformat()
                        check_in_time_fmt = last_in_log.timestamp.strftime("%I:%M:%S %p")
                else:
                    final_status = "CLAIMED_IDENTITY_VERIFIED"
                    decision_reason = "CHECK-IN VERIFIED"
                    db_status = "ACCESS_GRANTED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, claimed_score, action="CHECK-IN")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_in_time_iso = now_dt.isoformat()
                    check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")
            elif action == "CHECK-OUT":
                if not is_inside:
                    final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
                    decision_reason = "USER_NOT_INSIDE"
                    db_status = "ACCESS_DENIED"
                    saved_log = None
                else:
                    final_status = "CLAIMED_IDENTITY_VERIFIED"
                    decision_reason = "CHECK-OUT VERIFIED"
                    db_status = "ACCESS_GRANTED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, claimed_score, action="CHECK-OUT")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_out_time_iso = now_dt.isoformat()
                    check_out_time_fmt = now_dt.strftime("%I:%M:%S %p")

                    if last_in_log and last_in_log.timestamp:
                        check_in_time_iso = last_in_log.timestamp.isoformat()
                        check_in_time_fmt = last_in_log.timestamp.strftime("%I:%M:%S %p")
                        diff_seconds = max(0, int((now_dt - last_in_log.timestamp).total_seconds()))
                        hrs = diff_seconds // 3600
                        mins = (diff_seconds % 3600) // 60
                        secs = diff_seconds % 60
                        total_duration_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"
                    else:
                        check_in_time_iso = now_dt.isoformat()
                        check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")
                        total_duration_str = "00h 00m 00s"
            else:
                final_status = "CLAIMED_IDENTITY_VERIFIED"
                decision_reason = "DIAGNOSTIC VERIFIED"
                db_status = "ACCESS_GRANTED"
                saved_log = None
                if not is_dry:
                    saved_log = log_access_event(db, claimed_user.name, db_status, claimed_score, action=action)

        matched_user_id = claimed_user.id if biometrics_passed else None
        matched_user_name = claimed_user.name if biometrics_passed else f"Not Verified ({claimed_user.name})"

        print(f"[1:1 VERIFICATION] Claimed User: '{claimed_user.name}' (#{claimed_user.id}) | Score: {claimed_score:.4f} | Competitor: {strongest_other_score:.4f} | Margin: {identity_margin:.4f} | Action: {action} | Status: {final_status} ({decision_reason}) | Biometrics Passed: {biometrics_passed}")

        return {
            "face_detected": True,
            "left_eye_detected": payload.get("left_eye_detected", False),
            "right_eye_detected": payload.get("right_eye_detected", False),
            "claimed_user_id": claimed_user.id,
            "claimed_user_name": claimed_user.name,
            "template_ids_used": template_ids_used,
            "templates_checked": len(template_ids_used),
            "left_score": round(l_agg, 4) if left_vec else None,
            "right_score": round(r_agg, 4) if right_vec else None,
            "best_similarity": round(claimed_score, 4),
            "strongest_competing_score": round(strongest_other_score, 4),
            "identity_margin": round(identity_margin, 4),
            "left_eye_consistency": left_consistency if biometrics_passed else "FAIL",
            "right_eye_consistency": right_consistency if biometrics_passed else "FAIL",
            "best_match_user_id": matched_user_id,
            "best_match_user_name": matched_user_name,
            "diagnostic_threshold": RGB_1TO1_THRESHOLD,
            "min_identity_margin": MIN_IDENTITY_MARGIN,
            "final_diagnostic_status": final_status,
            "decision_reason": decision_reason,
            "biometric_verified": biometrics_passed,
            "check_in_time": check_in_time_iso,
            "check_in_time_formatted": check_in_time_fmt,
            "check_out_time": check_out_time_iso,
            "check_out_time_formatted": check_out_time_fmt,
            "total_duration": total_duration_str,
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
            "annotated_image": annotated_base64,
            "message": f"1:1 Verification Result: {final_status} ({decision_reason})"
        }

    except Exception as e:
        print("[1:1 Verification Exception]:", e)
        return {
            "face_detected": False,
            "claimed_user_id": claimed_user_id,
            "claimed_user_name": "Error",
            "best_similarity": 0.0,
            "strongest_competing_score": 0.0,
            "identity_margin": 0.0,
            "left_eye_consistency": "FAIL",
            "right_eye_consistency": "FAIL",
            "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
            "decision_reason": f"ERROR: {str(e)}",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
            "message": str(e)
        }
    finally:
        db.close()


def recognize_iris_v2_sequence_1to1_diagnostic(
    images_input,
    claimed_user_id: int,
    action: str = "CHECK-IN",
    dry_run: bool = False
):
    """
    Multi-Frame 1:1 Claimed Identity Verification (5-7 Frames) with Anti-False-Accept Guard.
    Requires at least 4 out of 5 valid frames (>= 80%) to score >= 0.80 AND margin >= 0.08.
    """
    db = SessionLocal()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RGB_1TO1_THRESHOLD = DEFAULT_IRIS_MATCH_THRESHOLD
    MIN_IDENTITY_MARGIN = DEFAULT_IRIS_MIN_IDENTITY_MARGIN
    is_dry = bool(dry_run) or (str(action).upper() in ["DIAGNOSTIC", "DRY-RUN", "DRY_RUN", "TEST"])

    try:
        claimed_user = db.query(User).filter(User.id == claimed_user_id).first()
        if not claimed_user:
            return {
                "sequence_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
                "claimed_user_id": claimed_user_id,
                "claimed_user_name": f"User ID #{claimed_user_id} Not Found",
                "decision_reason": f"CLAIMED_USER_NOT_FOUND: User ID #{claimed_user_id} does not exist",
                "biometric_verified": False,
                "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE"
            }

        all_recs = db.query(IrisEmbeddingV2).all()
        user_templates = {}
        for r in all_recs:
            if r.user_id not in user_templates:
                user_templates[r.user_id] = {"L": [], "R": []}
            user_templates[r.user_id][r.eye_side].append(r.embedding)

        frame_details = []
        valid_frames = 0
        passing_frames = 0
        claimed_scores_list = []
        competitor_scores_list = []
        last_annotated_base64 = None

        for idx, img_src in enumerate(images_input):
            if isinstance(img_src, str):
                img_np = cv2.cvtColor(cv2.imread(img_src), cv2.COLOR_BGR2RGB)
            elif isinstance(img_src, Image.Image):
                img_np = np.array(img_src.convert("RGB"))
            else:
                img_np = np.array(img_src)

            res = detect_and_crop_iris(img_np)
            payload = res.payload

            buffered = io.BytesIO()
            res[1].save(buffered, format="JPEG")
            last_annotated_base64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

            if not payload.get("face_detected"):
                frame_details.append({"frame": idx, "status": "NO_FACE", "score": 0.0, "margin": 0.0})
                continue

            left_iris_det = payload.get("left_iris_detected", False)
            right_iris_det = payload.get("right_iris_detected", False)

            if not (left_iris_det or right_iris_det):
                frame_details.append({"frame": idx, "status": "QUALITY_FAILED", "score": 0.0, "margin": 0.0})
                continue

            valid_frames += 1

            left_vec = None
            if left_iris_det and payload.get("left_eye_bbox"):
                lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
                left_crop = Image.fromarray(img_np[ly1:ly2, lx1:lx2])
                left_vec = extract_v2_embedding(left_crop)

            right_vec = None
            if right_iris_det and payload.get("right_eye_bbox"):
                rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
                right_crop = Image.fromarray(img_np[ry1:ry2, rx1:rx2])
                right_vec = extract_v2_embedding(right_crop)

            f_user_scores = {}
            for uid, t_dict in user_templates.items():
                f_l_sims = []
                if left_vec and t_dict["L"]:
                    lt = torch.tensor(left_vec, device=device).unsqueeze(0)
                    for emb in t_dict["L"]:
                        st = torch.tensor(emb, device=device).unsqueeze(0)
                        f_l_sims.append(float(F.cosine_similarity(lt, st, dim=1).item()))

                f_r_sims = []
                if right_vec and t_dict["R"]:
                    rt = torch.tensor(right_vec, device=device).unsqueeze(0)
                    for emb in t_dict["R"]:
                        st = torch.tensor(emb, device=device).unsqueeze(0)
                        f_r_sims.append(float(F.cosine_similarity(rt, st, dim=1).item()))

                fl_agg = float(np.mean(sorted(f_l_sims, reverse=True)[:2])) if f_l_sims else 0.0
                fr_agg = float(np.mean(sorted(f_r_sims, reverse=True)[:2])) if f_r_sims else 0.0
                f_user_scores[uid] = (fl_agg + fr_agg) / 2.0 if (f_l_sims and f_r_sims) else (fl_agg or fr_agg)

            f_claimed_score = f_user_scores.get(claimed_user.id, 0.0)
            f_other_scores = [sc for uid, sc in f_user_scores.items() if uid != claimed_user.id]
            f_strongest_other = float(max(f_other_scores)) if f_other_scores else 0.0
            f_margin = f_claimed_score - f_strongest_other

            claimed_scores_list.append(f_claimed_score)
            competitor_scores_list.append(f_strongest_other)

            if f_claimed_score >= RGB_1TO1_THRESHOLD and f_margin >= MIN_IDENTITY_MARGIN:
                passing_frames += 1
                frame_details.append({"frame": idx, "status": "PASSED", "score": round(f_claimed_score, 4), "margin": round(f_margin, 4)})
            else:
                frame_details.append({"frame": idx, "status": "FAILED", "score": round(f_claimed_score, 4), "margin": round(f_margin, 4)})

        if valid_frames == 0 or not claimed_scores_list:
            if not is_dry:
                log_access_event(db, claimed_user.name, "ACCESS_DENIED", 0.0, action=action)
            return {
                "face_detected": False,
                "claimed_user_id": claimed_user.id,
                "claimed_user_name": claimed_user.name,
                "best_similarity": 0.0,
                "strongest_competing_score": 0.0,
                "identity_margin": 0.0,
                "final_diagnostic_status": "CLAIMED_IDENTITY_NOT_VERIFIED",
                "decision_reason": "INSUFFICIENT_SEQUENCE_EVIDENCE",
                "biometric_verified": False,
                "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
                "annotated_image": last_annotated_base64
            }

        avg_claimed = float(np.mean(claimed_scores_list))
        avg_competing = float(np.mean(competitor_scores_list))
        seq_margin = avg_claimed - avg_competing

        pass_pct = (passing_frames / valid_frames * 100.0) if valid_frames > 0 else 0.0

        # Multi-Frame Decision Rules
        biometrics_passed = True
        if avg_claimed < RGB_1TO1_THRESHOLD:
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "SCORE_BELOW_THRESHOLD"
            biometrics_passed = False
        elif seq_margin < MIN_IDENTITY_MARGIN:
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "COMPETING_IDENTITY_TOO_CLOSE"
            biometrics_passed = False
        elif passing_frames < 4 or pass_pct < 80.0:
            final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
            decision_reason = "INSUFFICIENT_SEQUENCE_EVIDENCE"
            biometrics_passed = False

        check_in_time_iso = None
        check_in_time_fmt = None
        check_out_time_iso = None
        check_out_time_fmt = None
        total_duration_str = None

        if not biometrics_passed:
            db_status = "ACCESS_DENIED"
            saved_log = None
            if not is_dry:
                saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action=action)
        else:
            is_inside, last_in = is_user_currently_inside(db, claimed_user.name)

            if action == "CHECK-IN":
                if is_inside:
                    # RULE 2: USER IS ALREADY INSIDE -> REJECT DUPLICATE CHECK-IN!
                    final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
                    decision_reason = "USER_ALREADY_INSIDE"
                    db_status = "ACCESS_DENIED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action="CHECK-IN")
                else:
                    # RULE 1: USER IS OUTSIDE -> GRANT CHECK-IN & RECORD TRANSACTION!
                    final_status = "CLAIMED_IDENTITY_VERIFIED"
                    decision_reason = "CHECK-IN VERIFIED"
                    db_status = "ACCESS_GRANTED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action="CHECK-IN")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_in_time_iso = now_dt.isoformat()
                    check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")

            elif action == "CHECK-OUT":
                if not is_inside:
                    # RULE 3: USER IS NOT INSIDE -> REJECT CHECK-OUT!
                    final_status = "CLAIMED_IDENTITY_NOT_VERIFIED"
                    decision_reason = "USER_NOT_INSIDE"
                    db_status = "ACCESS_DENIED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action="CHECK-OUT")
                else:
                    # RULE 3: USER IS INSIDE -> GRANT CHECK-OUT & CLOSE SESSION!
                    final_status = "CLAIMED_IDENTITY_VERIFIED"
                    decision_reason = "CHECK-OUT VERIFIED"
                    db_status = "ACCESS_GRANTED"
                    saved_log = None
                    if not is_dry:
                        saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action="CHECK-OUT")
                    now_dt = saved_log.timestamp if (saved_log and saved_log.timestamp) else datetime.now()
                    check_out_time_iso = now_dt.isoformat()
                    check_out_time_fmt = now_dt.strftime("%I:%M:%S %p")

                    if last_in and last_in.timestamp:
                        check_in_time_iso = last_in.timestamp.isoformat()
                        check_in_time_fmt = last_in.timestamp.strftime("%I:%M:%S %p")
                        diff_seconds = int((now_dt - last_in.timestamp).total_seconds())
                        hrs = diff_seconds // 3600
                        mins = (diff_seconds % 3600) // 60
                        secs = diff_seconds % 60
                        total_duration_str = f"{hrs:02d}h {mins:02d}m {secs:02d}s"
                    else:
                        check_in_time_iso = now_dt.isoformat()
                        check_in_time_fmt = now_dt.strftime("%I:%M:%S %p")
                        total_duration_str = "00h 00m 00s"
            else:
                final_status = "CLAIMED_IDENTITY_VERIFIED"
                decision_reason = "DIAGNOSTIC VERIFIED"
                db_status = "ACCESS_GRANTED"
                saved_log = None
                if not is_dry:
                    saved_log = log_access_event(db, claimed_user.name, db_status, avg_claimed, action=action)

        matched_user_id = claimed_user.id if biometrics_passed else None
        matched_user_name = claimed_user.name if biometrics_passed else f"Not Verified ({claimed_user.name})"

        return {
            "face_detected": True,
            "left_eye_detected": True,
            "right_eye_detected": True,
            "claimed_user_id": claimed_user.id,
            "claimed_user_name": claimed_user.name,
            "templates_checked": len(user_templates.get(claimed_user.id, {}).get("L", [])) + len(user_templates.get(claimed_user.id, {}).get("R", [])),
            "total_frames_processed": len(images_input),
            "valid_frames_count": valid_frames,
            "passing_frames_count": passing_frames,
            "failing_frames_count": valid_frames - passing_frames,
            "best_similarity": round(avg_claimed, 4),
            "strongest_competing_score": round(avg_competing, 4),
            "identity_margin": round(seq_margin, 4),
            "left_eye_consistency": "PASS" if biometrics_passed else "FAIL",
            "right_eye_consistency": "PASS" if biometrics_passed else "FAIL",
            "best_match_user_id": matched_user_id,
            "best_match_user_name": matched_user_name,
            "frame_details": frame_details,
            "diagnostic_threshold": RGB_1TO1_THRESHOLD,
            "min_identity_margin": MIN_IDENTITY_MARGIN,
            "final_diagnostic_status": final_status,
            "decision_reason": decision_reason,
            "biometric_verified": biometrics_passed,
            "action": action,
            "check_in_time": check_in_time_iso,
            "check_in_time_formatted": check_in_time_fmt,
            "check_out_time": check_out_time_iso,
            "check_out_time_formatted": check_out_time_fmt,
            "total_duration": total_duration_str,
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry else "NONE",
            "annotated_image": last_annotated_base64,
            "message": f"1:1 Multi-Frame Result: {final_status} ({decision_reason})"
        }

    finally:
        db.close()