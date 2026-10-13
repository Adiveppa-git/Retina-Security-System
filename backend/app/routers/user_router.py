from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional, List, Any
from PIL import Image
import io

from app.services.user_management_service import (
    register_user,
    register_user_v2,
    get_all_users,
    delete_user
)

router = APIRouter(
    prefix="/user",
    tags=["User"]
)


@router.post("/add")
async def add_user(
    name: str = Form(...),
    email: Optional[str] = Form(None),
    files: List[UploadFile] = File(...)
):
    images = []
    for file_item in files:
        image_bytes = await file_item.read()
        if image_bytes:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(img)

    # Enroll into dedicated V2 embeddings table (iris_embeddings_v2) without touching V1 records
    res_v2 = register_user_v2(name=name, email=email, image_files=images)
    return {
        "status": res_v2.get("status", "error"),
        "message": res_v2.get("message"),
        "user_id": res_v2.get("user_id"),
        "templates_registered": res_v2.get("v2_templates_registered", 0)
    }


@router.get("/all")
def all_users():
    return get_all_users()


@router.delete("/{user_name}")
def remove_user(user_name: str):
    return delete_user(user_name)


# ---------------------------------------------------------------------------
# Face Biometric Sample Registration & Storage Endpoints
# ---------------------------------------------------------------------------
from app.services.face_recognition_service import (
    register_face_sample,
    get_user_face_samples,
    identify_user_by_face,
    verify_two_stage_identity,
    verify_two_stage_sequence_identity,
    DEFAULT_FACE_MATCH_THRESHOLD
)


@router.get("/audit_templates")
def audit_templates_endpoint():
    """
    Read-only audit for suspicious existing biometric templates.
    Never alters or deletes database records.
    """
    from app.services.template_audit_service import audit_enrolled_templates
    return audit_enrolled_templates(verbose=False)


@router.post("/{user_id}/face/register")
async def register_user_face(
    user_id: int,
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Registers a single face embedding sample for an existing user into MySQL
    face_embeddings table. Multiple samples are preserved without overwriting.
    """
    target_file = file or image
    if target_file is None:
        return {
            "status": "error",
            "error_code": "NO_FILE_PROVIDED",
            "message": "No face image file provided. Please attach a 'file' or 'image' form part.",
            "user_id": user_id,
            "sample_id": None
        }

    image_bytes = await target_file.read()
    if not image_bytes:
        return {
            "status": "error",
            "error_code": "EMPTY_FILE",
            "message": "Uploaded face image file is empty.",
            "user_id": user_id,
            "sample_id": None
        }

    result = register_face_sample(user_id=user_id, image_data=image_bytes)
    return result


@router.get("/{user_id}/face/samples")
def list_user_face_samples(user_id: int):
    """
    Retrieves metadata of all registered face samples for a user.
    """
    return get_user_face_samples(user_id)


@router.post("/face/identify")
@router.post("/identify_face")
async def identify_user_face_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Stage 1: 1:N Facial Identity Matching Endpoint.
    Accepts a live face image, detects face, extracts SFace 128-D embedding,
    compares against all enrolled face embeddings in MySQL, and returns
    the matching candidate_user_id if the best score meets the threshold.
    """
    target_file = file or image
    if target_file is None:
        return {
            "status": "error",
            "error_code": "NO_FILE_PROVIDED",
            "matched": False,
            "reason": "NO_FILE_PROVIDED",
            "message": "No face image file provided. Please attach 'file' or 'image' form part.",
            "candidate_user_id": None,
            "candidate_user_name": None,
            "face_score": 0.0
        }

    image_bytes = await target_file.read()
    if not image_bytes:
        return {
            "status": "error",
            "error_code": "EMPTY_FILE",
            "matched": False,
            "reason": "EMPTY_FILE",
            "message": "Uploaded face image file is empty.",
            "candidate_user_id": None,
            "candidate_user_name": None,
            "face_score": 0.0
        }

    result = identify_user_by_face(image=image_bytes)
    return result


def is_genuine_identity_identification_failure(res: dict) -> bool:
    """
    Checks if identity identification definitively completed and failed because
    no registered user was recognized (unregistered person / identity mismatch),
    distinguishing this from technical/camera/detection errors.
    """
    if res.get("failure_stage") != "IDENTITY_IDENTIFICATION":
        return False
    if res.get("status") == "error":
        return False
    face_st = res.get("face_stage") or {}
    if face_st.get("status") == "error":
        return False
    technical_errors = {
        "NO_FACE_DETECTED",
        "MULTIPLE_FACES_DETECTED",
        "EXTRACTION_FAILED",
        "EXTRACTION_ERROR",
        "INVALID_IMAGE",
        "CAMERA_ERROR",
        "NO_EYES_DETECTED",
        "LOW_QUALITY",
        "NO_FRAMES_PROVIDED",
    }
    reason = str(face_st.get("reason", "")).upper()
    if reason in technical_errors:
        return False
    return (
        face_st.get("matched") is False
        and face_st.get("candidate_user_id") is None
        and reason in {"IDENTITY_NOT_FOUND", "NO_REGISTERED_USERS"}
    )


@router.post("/verify_identity")
async def verify_identity_endpoint(
    file: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),
    files: List[UploadFile] = File(default=[]),
    iris_file: Optional[UploadFile] = File(None),
    action: str = Form("CHECK-IN"),
    dry_run: Any = Form(False)
):
    """
    Two-Stage Identity Verification Endpoint:
    Supports both single-frame and coordinated multi-frame sequence verification.
    Stage 1: 1:N Facial Identity Matching -> selects candidate_user_id.
    Stage 2: 1:1 Iris Verification against candidate_user_id templates ONLY.
    
    Safety:
    - If dry_run is True, no access_logs or hardware actions (ESP32/GPIO) are triggered.
    - If Stage 1 fails, iris stage is bypassed (executed: False).
    """
    is_dry_run = str(dry_run).lower() in ["true", "1", "yes"] if isinstance(dry_run, str) else bool(dry_run)

    # Check for coordinated sequence upload (multiple frames)
    if files and len(files) > 1:
        frame_bytes_list = []
        for f_item in files:
            b = await f_item.read()
            if b:
                frame_bytes_list.append(b)

        if frame_bytes_list:
            result = verify_two_stage_sequence_identity(
                frames_input=frame_bytes_list,
                action=action,
                dry_run=is_dry_run
            )
            # Hardware Access Status Dispatcher: Reacts ONLY to the FINAL decision
            if not is_dry_run:
                from app.routers.iris_router import send_to_esp32
                if result.get("final_decision") == "VERIFIED":
                    hw_res = send_to_esp32("A")
                    if hw_res.get("status") == "success":
                        result["hardware_action"] = "SENT_A_AUTHORIZED"
                    else:
                        result["hardware_action"] = f"HARDWARE_COMM_ERROR: {hw_res.get('error', 'Unknown Error')}"
                        result["hardware_error"] = hw_res.get("error")
                elif result.get("failure_stage") == "BIOMETRIC_VERIFICATION" or is_genuine_identity_identification_failure(result):
                    hw_res = send_to_esp32("U")
                    if hw_res.get("status") == "success":
                        result["hardware_action"] = "SENT_U_UNAUTHORIZED"
                    else:
                        result["hardware_action"] = f"HARDWARE_COMM_ERROR: {hw_res.get('error', 'Unknown Error')}"
                        result["hardware_error"] = hw_res.get("error")
                else:
                    result["hardware_action"] = f"NO_COMMAND_{result.get('failure_stage', 'REJECTED')}"
            else:
                result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"

            return result

    # Single-frame fallback (or files containing 1 frame)
    target_file = file or image or (files[0] if files and len(files) == 1 else None)
    if target_file is None:
        return {
            "status": "error",
            "error_code": "NO_FILE_PROVIDED",
            "face_stage": {"matched": False},
            "iris_stage": {"executed": False},
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry_run else "NO_COMMAND_NO_FILE"
        }

    image_bytes = await target_file.read()
    if not image_bytes:
        return {
            "status": "error",
            "error_code": "EMPTY_FILE",
            "face_stage": {"matched": False},
            "iris_stage": {"executed": False},
            "final_decision": "REJECTED",
            "failure_stage": "IDENTITY_IDENTIFICATION",
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry_run else "NO_COMMAND_EMPTY_FILE"
        }

    iris_bytes = None
    if iris_file is not None:
        iris_bytes = await iris_file.read()

    result = verify_two_stage_identity(
        face_image_input=image_bytes,
        iris_image_input=iris_bytes if iris_bytes else image_bytes,
        action=action,
        dry_run=is_dry_run
    )

    # Hardware Access Status Dispatcher: Reacts ONLY to the FINAL decision
    if not is_dry_run:
        from app.routers.iris_router import send_to_esp32
        if result.get("final_decision") == "VERIFIED":
            hw_res = send_to_esp32("A")
            if hw_res.get("status") == "success":
                result["hardware_action"] = "SENT_A_AUTHORIZED"
            else:
                result["hardware_action"] = f"HARDWARE_COMM_ERROR: {hw_res.get('error', 'Unknown Error')}"
                result["hardware_error"] = hw_res.get("error")
        elif result.get("failure_stage") == "BIOMETRIC_VERIFICATION" or is_genuine_identity_identification_failure(result):
            hw_res = send_to_esp32("U")
            if hw_res.get("status") == "success":
                result["hardware_action"] = "SENT_U_UNAUTHORIZED"
            else:
                result["hardware_action"] = f"HARDWARE_COMM_ERROR: {hw_res.get('error', 'Unknown Error')}"
                result["hardware_error"] = hw_res.get("error")
        else:
            result["hardware_action"] = f"NO_COMMAND_{result.get('failure_stage', 'REJECTED')}"
    else:
        result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"

    return result