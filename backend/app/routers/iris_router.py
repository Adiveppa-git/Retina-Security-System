import io
import os
import time
from typing import List
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image
import serial

from app.services.iris_recognition_service import (
    recognize_iris_v2_diagnostic,
    recognize_iris_v2_sequence_diagnostic,
    recognize_iris_v2_1to1_diagnostic,
    recognize_iris_v2_sequence_1to1_diagnostic
)

ESP32_PORT = "COM9"
ESP32_BAUD = 115200


def resolve_hardware_command(result):
    """
    Hardware output safety gate:
    - Only explicit verified matches send 'A'.
    - Actual biometric mismatch sends 'U'.
    - Detection failures, camera failures, and system errors must not claim
      the person is unauthorized or send any command to the ESP32.
    """
    if not result:
        return None, "NO_COMMAND_NO_AUTH_RESULT"

    status = str(result.get("final_diagnostic_status") or "").upper()
    reason = str(result.get("decision_reason") or "").upper()

    if status in {"CLAIMED_IDENTITY_VERIFIED", "V2_DIAGNOSTIC_MATCH"}:
        return "A", "SENT_A_AUTHORIZED"

    if status in {"CLAIMED_IDENTITY_NOT_VERIFIED", "V2_DIAGNOSTIC_UNMATCHED"}:
        if any(token in reason for token in [
            "NO_FACE",
            "QUALITY",
            "SPOOF",
            "ERROR",
            "NOT_FOUND",
            "USER_ALREADY_INSIDE",
            "USER_NOT_INSIDE",
            "INSUFFICIENT_SEQUENCE",
            "NO_VALID_EYES",
            "SYSTEM_ERROR"
        ]):
            return None, "NO_COMMAND_AUTHENTICATION_FAILED"
        return "U", "SENT_U_UNAUTHORIZED"

    if "ERROR" in reason or "NOT_FOUND" in reason or "NO_FACE" in reason:
        return None, "NO_COMMAND_SYSTEM_ERROR"

    return None, "NO_COMMAND_NO_AUTH_RESULT"


def send_to_esp32(command: str) -> dict:
    """
    Sends command to ESP32 over USB Serial (COM9, 115200 baud).
    - 'A': AUTHORIZED (Cancel alarm immediately: buzzer OFF, red LED GPIO 5 OFF, normal LED GPIO 2 ON for 2.5s)
    - 'U': UNAUTHORIZED (Normal LED OFF, red LED GPIO 5 ON, repeating buzzer alarm on GPIO 4 for 15s)
    Configures dtr=False and rts=False to prevent resetting ESP32 upon port connection.
    Waits for handshake acknowledgment to ensure command processing before closing.
    Returns dict with status, command, and ack/error details.
    """
    if os.getenv("DISABLE_ESP32_COMMANDS", "").lower() in ["true", "1", "yes"]:
        print(f"[ESP32 SAFEGUARD] Blocked command '{command}' because DISABLE_ESP32_COMMANDS is active.")
        return {"status": "suppressed", "command": command, "message": "DISABLE_ESP32_COMMANDS is active"}

    esp32 = None
    try:
        esp32 = serial.Serial()
        esp32.port = ESP32_PORT
        esp32.baudrate = ESP32_BAUD
        esp32.dtr = False
        esp32.rts = False
        esp32.timeout = 0.5
        esp32.open()
        time.sleep(0.08)

        cmd_payload = f"{command.strip().upper()}\n".encode("utf-8")
        esp32.write(cmd_payload)
        esp32.flush()
        ack = esp32.readline().decode("utf-8", errors="ignore").strip()
        if not ack:
            ack = esp32.readline().decode("utf-8", errors="ignore").strip()
        print(f"[ESP32] Sent: '{command}' to {ESP32_PORT} | Ack: '{ack}'")
        return {"status": "success", "command": command, "ack": ack, "port": ESP32_PORT}
    except Exception as e:
        err_msg = str(e)
        print(f"[ESP32 ERROR] Failed to send command '{command}' to {ESP32_PORT}: {err_msg}")
        return {"status": "error", "command": command, "error": err_msg, "port": ESP32_PORT}
    finally:
        if esp32 and esp32.is_open:
            try:
                esp32.close()
            except Exception:
                pass


router = APIRouter(
    prefix="/iris",
    tags=["Iris"]
)


@router.get("/hardware/status")
def hardware_status():
    """
    Checks connection status of ESP32 on COM9 without transmitting activating commands.
    """
    import serial.tools.list_ports
    ports = [p.device for p in serial.tools.list_ports.comports()]
    is_connected = ESP32_PORT in ports
    return {
        "port": ESP32_PORT,
        "baud": ESP32_BAUD,
        "connected": is_connected,
        "available_ports": ports
    }


@router.post("/hardware/test")
def hardware_test(command: str = Form(...)):
    """
    Explicitly initiated hardware test endpoint.
    Safely tests ESP32 indicators ('A' or 'U') without granting user authorization
    and without mutating access logs or check-in/check-out states.
    """
    cmd = command.strip().upper()
    if cmd not in ["A", "U"]:
        return {
            "status": "error",
            "error_code": "INVALID_COMMAND",
            "message": f"Invalid command '{command}'. Allowed: 'A' (Authorized/Cancel) or 'U' (Unauthorized/Alarm)."
        }

    hw_res = send_to_esp32(cmd)
    if hw_res.get("status") == "success":
        return {
            "status": "success",
            "mode": "HARDWARE_TEST",
            "command": cmd,
            "ack": hw_res.get("ack"),
            "hardware_action": f"TEST_{cmd}_EXECUTED",
            "message": f"Hardware test command '{cmd}' executed on {ESP32_PORT}. Ack: '{hw_res.get('ack')}'. Zero database mutations."
        }
    else:
        return {
            "status": "error",
            "mode": "HARDWARE_TEST",
            "command": cmd,
            "error_code": "HARDWARE_COMM_ERROR",
            "hardware_action": f"HARDWARE_TEST_FAILED",
            "message": f"Failed to communicate with ESP32 on {ESP32_PORT}: {hw_res.get('error')}"
        }


@router.post("/verify")
async def verify_iris(
    file: UploadFile = File(...),
    dry_run: bool = Form(False)
):
    start_time = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = recognize_iris_v2_diagnostic(image)

    if dry_run:
        result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"
        print("[SAFE DIAGNOSTIC] /verify complete. Hardware command SUPPRESSED.")
    else:
        command, action_label = resolve_hardware_command(result)
        if command == "A":
            send_to_esp32("A")
            result["hardware_action"] = action_label
        elif command == "U":
            send_to_esp32("U")
            result["hardware_action"] = action_label
        else:
            result["hardware_action"] = action_label
            print(f"[SAFE GUARD] No ESP32 command sent for status {result.get('final_diagnostic_status')} reason={result.get('decision_reason')}")

    end_time = time.time()
    result["response_time"] = f"{round(end_time - start_time, 3)} sec"
    result["camera_source"] = "Zebronics USB Webcam"
    result["camera_resolution"] = f"{image.width}x{image.height}"

    return result


@router.post("/verify_1to1")
async def verify_iris_1to1(
    file: UploadFile = File(...),
    claimed_user_id: int = Form(...),
    action: str = Form("CHECK-IN"),
    dry_run: bool = Form(False)
):
    start_time = time.time()

    is_dry_run = bool(dry_run) or (str(action).upper() in ["DIAGNOSTIC", "DRY-RUN", "DRY_RUN", "TEST"])

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    result = recognize_iris_v2_1to1_diagnostic(image, claimed_user_id, action, dry_run=is_dry_run)

    # In safe diagnostic mode, do NOT send commands to ESP32
    if is_dry_run:
        result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"
        print(f"[SAFE DIAGNOSTIC] 1:1 Verification complete for user #{claimed_user_id} (Status: {result.get('final_diagnostic_status')}). Hardware action SUPPRESSED. No command sent to ESP32.")
    else:
        command, action_label = resolve_hardware_command(result)
        if command == "A":
            send_to_esp32("A")
            result["hardware_action"] = action_label
        elif command == "U":
            send_to_esp32("U")
            result["hardware_action"] = action_label
        else:
            result["hardware_action"] = action_label
            print(f"[SAFE GUARD] No ESP32 command sent for status {result.get('final_diagnostic_status')} reason={result.get('decision_reason')}")

    end_time = time.time()
    result["response_time"] = f"{round(end_time - start_time, 3)} sec"
    result["camera_source"] = "Zebronics USB Webcam (Single-Frame 1:1)"
    result["camera_resolution"] = f"{image.width}x{image.height}"

    return result


@router.post("/verify_sequence_1to1")
async def verify_iris_sequence_1to1(
    files: List[UploadFile] = File(...),
    claimed_user_id: int = Form(...),
    action: str = Form("CHECK-IN"),
    dry_run: bool = Form(False)
):
    start_time = time.time()

    is_dry_run = bool(dry_run) or (str(action).upper() in ["DIAGNOSTIC", "DRY-RUN", "DRY_RUN", "TEST"])

    pil_images = []
    for file in files:
        bytes_data = await file.read()
        img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        pil_images.append(img)

    result = recognize_iris_v2_sequence_1to1_diagnostic(pil_images, claimed_user_id, action, dry_run=is_dry_run)

    # In safe diagnostic mode, do NOT send commands to ESP32
    if is_dry_run:
        result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"
        print(f"[SAFE DIAGNOSTIC] Sequence Verification complete for user #{claimed_user_id} (Status: {result.get('final_diagnostic_status')}). Hardware action SUPPRESSED. No command sent to ESP32.")
    else:
        command, action_label = resolve_hardware_command(result)
        if command == "A":
            send_to_esp32("A")
            result["hardware_action"] = action_label
        elif command == "U":
            send_to_esp32("U")
            result["hardware_action"] = action_label
        else:
            result["hardware_action"] = action_label
            print(f"[SAFE GUARD] No ESP32 command sent for status {result.get('final_diagnostic_status')} reason={result.get('decision_reason')}")

    end_time = time.time()
    result["response_time"] = f"{round(end_time - start_time, 3)} sec"
    result["camera_source"] = "Zebronics USB Webcam (5-7 Frame 1:1 Sequence)"
    result["camera_resolution"] = f"{pil_images[0].width}x{pil_images[0].height}" if pil_images else "UNKNOWN"

    return result


@router.get("/camera/status")
def get_camera_status():
    """
    Checks the status of the primary Zebronics USB Webcam capture source.
    """
    from app.services.camera_interface import get_primary_camera_source
    cam = get_primary_camera_source()
    ok, frame, meta = cam.get_frame()
    cam.release()
    return {
        "camera_online": ok,
        "camera_source": "Zebronics USB Webcam",
        "metadata": meta
    }


@router.post("/verify_camera_1to1")
def verify_camera_1to1(
    claimed_user_id: int = Form(...),
    action: str = Form("CHECK-IN"),
    dry_run: bool = Form(False)
):
    """
    Captures a frame directly from the primary Zebronics USB Webcam source,
    executes 1:1 biometric iris recognition, and triggers ESP32 access control.
    """
    is_dry_run = bool(dry_run) or (str(action).upper() in ["DIAGNOSTIC", "DRY-RUN", "DRY_RUN", "TEST"])

    from app.services.camera_interface import get_primary_camera_source
    cam = get_primary_camera_source()
    ok, frame_rgb, meta = cam.get_frame()
    cam.release()

    if not ok or frame_rgb is None:
        return {
            "status": "error",
            "error": meta.get("error", "ZEBRONICS_CAMERA_DISCONNECTED"),
            "message": "Zebronics USB Webcam is unavailable or disconnected.",
            "camera_source": "Zebronics USB Webcam",
            "camera_metadata": meta,
            "hardware_action": "SUPPRESSED_DIAGNOSTIC" if is_dry_run else "NO_COMMAND_CAMERA_ERROR"
        }

    start_time = time.time()
    image = Image.fromarray(frame_rgb)
    result = recognize_iris_v2_1to1_diagnostic(image, claimed_user_id, action, dry_run=is_dry_run)

    if is_dry_run:
        result["hardware_action"] = "SUPPRESSED_DIAGNOSTIC"
        print(f"[SAFE DIAGNOSTIC] Camera 1:1 Verification complete for user #{claimed_user_id} (Status: {result.get('final_diagnostic_status')}). Hardware action SUPPRESSED. No command sent to ESP32.")
    else:
        command, action_label = resolve_hardware_command(result)
        if command == "A":
            send_to_esp32("A")
            result["hardware_action"] = action_label
        elif command == "U":
            send_to_esp32("U")
            result["hardware_action"] = action_label
        else:
            result["hardware_action"] = action_label
            print(f"[SAFE GUARD] No ESP32 command sent for status {result.get('final_diagnostic_status')} reason={result.get('decision_reason')}")

    end_time = time.time()
    result["response_time"] = f"{round(end_time - start_time, 3)} sec"
    result["camera_source"] = "Zebronics USB Webcam (Direct Hardware Capture)"
    result["camera_resolution"] = f"{image.width}x{image.height}"
    result["camera_metadata"] = meta

    return result



@router.get("/active_session/{user_id}")
def get_active_session_by_user(user_id: int):
    from app.database.connection import SessionLocal
    from app.models.user_model import User, AccessLog
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {"active": False, "message": "User not found"}

        last_granted = (
            db.query(AccessLog)
            .filter(AccessLog.user_name == user.name, AccessLog.status == "ACCESS_GRANTED")
            .order_by(AccessLog.timestamp.desc(), AccessLog.id.desc())
            .first()
        )

        if last_granted and (last_granted.action or "CHECK-IN") == "CHECK-IN":
            return {
                "active": True,
                "user_id": user.id,
                "user_name": user.name,
                "check_in_time": last_granted.timestamp.isoformat() if last_granted.timestamp else None,
                "check_in_formatted": last_granted.timestamp.strftime("%I:%M:%S %p") if last_granted.timestamp else "N/A",
                "status": "INSIDE"
            }
        return {"active": False, "user_id": user.id, "user_name": user.name}
    except Exception as e:
        return {"active": False, "error": str(e)}
    finally:
        db.close()


@router.get("/active_session")
def get_any_active_session():
    from app.database.connection import SessionLocal
    from app.models.user_model import User, AccessLog
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            last_granted = (
                db.query(AccessLog)
                .filter(AccessLog.user_name == user.name, AccessLog.status == "ACCESS_GRANTED")
                .order_by(AccessLog.timestamp.desc(), AccessLog.id.desc())
                .first()
            )
            if last_granted and (last_granted.action or "CHECK-IN") == "CHECK-IN":
                return {
                    "active": True,
                    "user_id": user.id,
                    "user_name": user.name,
                    "check_in_time": last_granted.timestamp.isoformat() if last_granted.timestamp else None,
                    "check_in_formatted": last_granted.timestamp.strftime("%I:%M:%S %p") if last_granted.timestamp else "N/A",
                    "status": "INSIDE"
                }
        return {"active": False}
    except Exception as e:
        return {"active": False, "error": str(e)}
    finally:
        db.close()


@router.get("/active_sessions")
def get_all_active_sessions():
    from app.database.connection import SessionLocal
    from app.models.user_model import User, AccessLog
    from sqlalchemy import func
    db = SessionLocal()
    try:
        users = db.query(User).all()
        active_list = []
        for user in users:
            last_granted = (
                db.query(AccessLog)
                .filter(func.lower(AccessLog.user_name) == func.lower(user.name), AccessLog.status == "ACCESS_GRANTED")
                .order_by(AccessLog.timestamp.desc(), AccessLog.id.desc())
                .first()
            )
            if last_granted and (last_granted.action or "CHECK-IN").upper() == "CHECK-IN":
                active_list.append({
                    "user_id": user.id,
                    "user_name": user.name,
                    "check_in_time": last_granted.timestamp.isoformat() if last_granted.timestamp else None,
                    "check_in_formatted": last_granted.timestamp.strftime("%I:%M:%S %p") if last_granted.timestamp else "N/A",
                    "status": "INSIDE"
                })
        return {"active_users": active_list, "total_inside": len(active_list)}
    except Exception as e:
        return {"active_users": [], "total_inside": 0, "error": str(e)}
    finally:
        db.close()


@router.get("/completed_visits")
def get_completed_visits():
    """
    Returns completed paired check-in and check-out visits from MySQL access logs.
    """
    from app.database.connection import SessionLocal
    from app.services.dashboard_service import get_completed_visits_data
    db = SessionLocal()
    try:
        visits = get_completed_visits_data(db, limit=20)
        return {"completed_visits": visits, "total": len(visits)}
    except Exception as e:
        return {"completed_visits": [], "total": 0, "error": str(e)}
    finally:
        db.close()