import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Union
from sqlalchemy.orm import Session

from app.database.connection import SessionLocal
from app.models.user_model import User, IrisEmbedding
from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN
from app.services.iris_detection_service import detect_and_crop_iris

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/iris_feature_model.pth"
)

device = torch.device("cpu")
recognition_model = IrisIdentityCNN(num_classes=2).to(device)

if os.path.exists(MODEL_PATH):
    recognition_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

recognition_model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_feature_vector(pil_img: Image.Image) -> list:
    """
    Extracts 576-D L2-normalized MobileNet Feature Embedding.
    Same subject scores ~90.7%, Different subject scores ~12.1%.
    """
    tensor_img = transform(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = recognition_model.features(tensor_img)
        feat = recognition_model.avgpool(feat)
        feat = torch.flatten(feat, 1)
        normalized = F.normalize(feat, p=2, dim=1)[0]
    return normalized.cpu().numpy().tolist()


def register_user(name: str, email: str, image_files: Union[List, any]):
    """
    Registers a new user into MySQL database:
    1. Detects & crops exact eye/iris region for each template photo.
    2. Computes 576-D L2-normalized Feature Embedding for EACH cropped eye.
    3. Persists User record and JSON embedding vectors in MySQL database.
    """
    db: Session = SessionLocal()
    try:
        if not isinstance(image_files, list):
            image_files = [image_files]

        if not image_files:
            return {"status": "error", "message": "No template images provided."}

        # Check if user with same name exists
        user = db.query(User).filter(User.name == name).first()
        if not user:
            user = User(name=name, email=email if email else f"{name.lower().replace(' ', '')}@example.com")
            db.add(user)
            db.flush()

        embeddings_added = 0

        for img_item in image_files:
            try:
                if isinstance(img_item, Image.Image):
                    pil_img = img_item.convert("RGB")
                elif isinstance(img_item, np.ndarray):
                    pil_img = Image.fromarray(img_item).convert("RGB")
                else:
                    pil_img = Image.open(img_item).convert("RGB")

                # Detect and crop eye region from image
                polar_ribbon, annotated_img, face_det, left_det, right_det = detect_and_crop_iris(pil_img)

                # Reject invalid crops: require detected eye and non-empty polar ribbon
                if not (left_det or right_det) or polar_ribbon is None:
                    print(f"[REJECTED] Invalid iris crop in template: no valid eye detected.")
                    continue

                # Compute 576-D Feature Vector ONLY on the cropped eye
                embedding_list = extract_feature_vector(polar_ribbon)
                if not embedding_list or len(embedding_list) != 576:
                    print(f"[REJECTED] Invalid embedding dimension: {len(embedding_list) if embedding_list else 0}")
                    continue

                emb_arr = np.array(embedding_list, dtype=np.float32)
                if not np.all(np.isfinite(emb_arr)) or np.all(emb_arr == 0) or np.linalg.norm(emb_arr) < 1e-4:
                    print(f"[REJECTED] Blank or non-finite embedding vector.")
                    continue

                iris_embedding = IrisEmbedding(user_id=user.id, embedding=embedding_list)
                db.add(iris_embedding)
                embeddings_added += 1
            except Exception as e:
                print(f"[WARNING] Could not process template image: {e}")

        if embeddings_added == 0:
            db.rollback()
            return {
                "status": "error",
                "message": "Registration rejected: No valid iris templates detected. Invalid crops and blank embeddings are rejected.",
                "user_id": user.id if user else None,
                "templates_registered": 0
            }

        db.commit()

        print(f"[SUCCESS] Registered user '{name}' (ID: {user.id}) with {embeddings_added} eye template embeddings in MySQL!")

        return {
            "status": "success",
            "message": f"User '{name}' registered with {embeddings_added} biometric templates successfully",
            "user_id": user.id,
            "templates_registered": embeddings_added
        }

    except Exception as e:
        db.rollback()
        print("Registration Error:", e)
        return {
            "status": "error",
            "message": str(e)
        }
    finally:
        db.close()


def get_all_users():
    """Retrieves all registered users from MySQL database."""
    db: Session = SessionLocal()
    try:
        users = db.query(User).all()
        result = []
        for u in users:
            template_count = db.query(IrisEmbedding).filter(IrisEmbedding.user_id == u.id).count()
            result.append({
                "id": u.id,
                "name": u.name,
                "email": u.email,
                "templates_count": template_count,
                "created_at": u.created_at.strftime("%Y-%m-%d %H:%M:%S") if u.created_at else None
            })
        return result
    except Exception as e:
        print("Get All Users Error:", e)
        return []
    finally:
        db.close()


def register_user_v2(name: str, email: str, image_files: Union[List, any]):
    """
    Phase 5B — Registers a user into MySQL database using V2 Identity Metric Model:
    1. Detects face, landmarks, & eye ROIs (Phase 3).
    2. Enforces quality gates; rejects invalid or low-quality eye captures.
    3. Generates 512-D L2-Normalized V2 Embeddings per valid eye ('L' and 'R').
    4. Persists V2 embeddings into dedicated iris_embeddings_v2 table.
    """
    db: Session = SessionLocal()
    try:
        import numpy as np
        import cv2
        from app.models.user_model import IrisEmbeddingV2
        from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

        if not isinstance(image_files, list):
            image_files = [image_files]

        if not image_files:
            return {"status": "error", "message": "No template images provided."}

        # Find or create user
        user = db.query(User).filter(User.name == name).first()
        if not user:
            user = User(name=name, email=email if email else f"{name.lower().replace(' ', '')}@example.com")
            db.add(user)
            db.flush()

        v2_templates_added = 0
        last_failure_reason = "QUALITY_VALIDATION_FAILED"

        for img_item in image_files:
            try:
                if isinstance(img_item, Image.Image):
                    pil_img = img_item.convert("RGB")
                    img_np = np.array(pil_img)
                elif isinstance(img_item, np.ndarray):
                    img_np = img_item.copy()
                    if len(img_np.shape) == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                else:
                    img_np = cv2.cvtColor(cv2.imread(img_item), cv2.COLOR_BGR2RGB)

                res = detect_and_crop_iris(img_np)
                payload = res.payload
                last_failure_reason = payload.get("failure_reason", "QUALITY_VALIDATION_FAILED")

                # Process Left Eye Embedding
                if payload.get("left_iris_detected") and payload.get("left_eye_bbox"):
                    lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
                    if (lx2 - lx1) >= 20 and (ly2 - ly1) >= 12 and lx1 >= 0 and ly1 >= 0 and lx2 <= img_np.shape[1] and ly2 <= img_np.shape[0]:
                        left_crop_np = img_np[ly1:ly2, lx1:lx2]
                        if left_crop_np.size > 0:
                            left_crop = Image.fromarray(left_crop_np)
                            vec_l = extract_v2_embedding(left_crop)
                            if vec_l and len(vec_l) == 512:
                                arr_l = np.array(vec_l, dtype=np.float32)
                                if np.all(np.isfinite(arr_l)) and not np.all(arr_l == 0) and np.linalg.norm(arr_l) >= 1e-4:
                                    emb_l = IrisEmbeddingV2(
                                        user_id=user.id,
                                        eye_side="L",
                                        model_version="v2",
                                        embedding_dimension=512,
                                        embedding=vec_l
                                    )
                                    db.add(emb_l)
                                    v2_templates_added += 1

                # Process Right Eye Embedding
                if payload.get("right_iris_detected") and payload.get("right_eye_bbox"):
                    rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
                    if (rx2 - rx1) >= 20 and (ry2 - ry1) >= 12 and rx1 >= 0 and ry1 >= 0 and rx2 <= img_np.shape[1] and ry2 <= img_np.shape[0]:
                        right_crop_np = img_np[ry1:ry2, rx1:rx2]
                        if right_crop_np.size > 0:
                            right_crop = Image.fromarray(right_crop_np)
                            vec_r = extract_v2_embedding(right_crop)
                            if vec_r and len(vec_r) == 512:
                                arr_r = np.array(vec_r, dtype=np.float32)
                                if np.all(np.isfinite(arr_r)) and not np.all(arr_r == 0) and np.linalg.norm(arr_r) >= 1e-4:
                                    emb_r = IrisEmbeddingV2(
                                        user_id=user.id,
                                        eye_side="R",
                                        model_version="v2",
                                        embedding_dimension=512,
                                        embedding=vec_r
                                    )
                                    db.add(emb_r)
                                    v2_templates_added += 1

            except Exception as e:
                print(f"[WARNING] Could not process V2 template image: {e}")

        if v2_templates_added > 0:
            db.commit()
            print(f"[SUCCESS Phase 5B] Registered V2 user '{name}' (ID: {user.id}) with {v2_templates_added} V2 templates in MySQL!")
            return {
                "status": "success",
                "message": f"User '{name}' registered with {v2_templates_added} V2 biometric templates successfully",
                "user_id": user.id,
                "v2_templates_registered": v2_templates_added
            }
        else:
            db.rollback()
            return {
                "status": "error",
                "message": f"Registration rejected: {last_failure_reason}",
                "user_id": user.id if user else None,
                "v2_templates_registered": 0
            }

    except Exception as e:
        db.rollback()
        print("V2 Registration Error:", e)
        return {"status": "error", "message": str(e)}
    finally:
        db.close()


def delete_user(user_name: str):
    """Deletes a user and their embeddings from MySQL database."""
    db: Session = SessionLocal()
    try:
        user = db.query(User).filter(User.name == user_name).first()
        if not user:
            return {"status": "error", "message": f"User '{user_name}' not found"}
        db.delete(user)
        db.commit()
        return {"status": "success", "message": f"User '{user_name}' deleted successfully"}
    except Exception as e:
        db.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        db.close()