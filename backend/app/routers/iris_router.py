from fastapi import APIRouter, UploadFile, File
from backend.app.services.anti_spoof_service import check_spoof
from backend.app.services.iris_detection_service import detect_iris
from backend.app.services.iris_recognition_service import recognize_iris
from backend.app.services.hardware_service import hardware_action
from backend.app.services.log_service import save_log

import io
import time
from PIL import Image

router = APIRouter(
    prefix="/iris",
    tags=["Iris"]
)


@router.post("/verify")
async def verify_iris(file: UploadFile = File(...)):

    # Start timer
    start_time = time.time()

    # Read uploaded image
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    # ---------------------------------------
    # Step 1 : Anti-Spoofing
    # ---------------------------------------
    if not check_spoof(image):

        end_time = time.time()
        response_time = f"{round(end_time - start_time, 3)} sec"

        save_log(
            person="Unknown",
            similarity=0.0,
            status="Fake Iris",
            response_time=response_time
        )

        return {
            "status": "Fake Iris Detected",
            "person": "Unknown",
            "similarity": 0.0,
            "response_time": response_time,
            "hardware": {
                "door": "Locked",
                "led": "Red",
                "buzzer": "On"
            }
        }

    # ---------------------------------------
    # Step 2 : Iris Detection
    # ---------------------------------------
    iris = detect_iris(image)
    result = recognize_iris(iris)

    # ---------------------------------------
    # Step 3 : Hardware Simulation
    # ---------------------------------------
    hardware = hardware_action(result["person"])

    # Stop timer
    end_time = time.time()
    response_time = f"{round(end_time - start_time, 3)} sec"

    # ---------------------------------------
    # Step 4 : Authorization Status
    # ---------------------------------------
    if result["person"] == "Unknown":
        status = "Access Denied"
    else:
        status = "Success"

    # ---------------------------------------
    # Step 5 : Save Log
    # ---------------------------------------
    save_log(
        person=result["person"],
        similarity=result["similarity"],
        status=status,
        response_time=response_time
    )

    # ---------------------------------------
    # Step 6 : Return Response
    # ---------------------------------------
    return {
        "status": status,
        "person": result["person"],
        "similarity": result["similarity"],
        "response_time": response_time,
        "hardware": hardware
    }