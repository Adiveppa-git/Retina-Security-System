import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.camera_interface import ZebronicsUSBWebcamSource, get_primary_camera_source
from backend.app.services.iris_detection_service import detect_and_crop_iris
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding

def test_zebronics_full_integration():
    print("==================================================")
    print(" TESTING ZEBRONICS USB WEBCAM INTEGRATION")
    print("==================================================")

    # 1. Primary Camera Source Acquisition
    cam = get_primary_camera_source(camera_index=1)
    ok, frame, meta = cam.get_frame()
    cam.release()

    print(f"\n1. Frame Acquisition Status: {ok}")
    print(f"   Metadata: {meta}")

    assert ok, f"Zebronics camera capture failed: {meta}"
    assert frame is not None, "Captured frame is None"
    assert meta["source"] == "Zebronics USB Webcam", f"Unexpected source: {meta['source']}"
    assert meta["color_format"] == "RGB", f"Unexpected color format: {meta['color_format']}"
    
    # 2. Quality & Exposure Validation
    mean_brightness = float(np.mean(frame))
    std_brightness = float(np.std(frame))
    print(f"\n2. Frame Quality & Exposure:")
    print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
    print(f"   Channels: {frame.shape[2]}")
    print(f"   Mean Brightness: {mean_brightness:.2f} (Target: > 40.0)")
    print(f"   Brightness Std Dev: {std_brightness:.2f}")

    assert mean_brightness > 40.0, f"Frame is too dark/black: mean brightness {mean_brightness}"
    print("   >>> FRAME IS NOT BLACK/DARK! EXPOSURE CONVERGED PROPERLY!")

    # 3. Downstream AI Preprocessing & Ribbon Localization
    res = detect_and_crop_iris(frame)
    ribbon = res[0]
    print(f"\n3. Downstream AI Detection:")
    print(f"   Normalized Ribbon: {ribbon.size if ribbon else 'None'}")
    print(f"   Face Detected: {res[2]}")
    print(f"   Left Eye Detected: {res[3]}")
    print(f"   Right Eye Detected: {res[4]}")

    if ribbon:
        emb = extract_v2_embedding(ribbon)
        print(f"   Vector Dimension: {len(emb)}-D")
        assert len(emb) == 512, f"Expected 512-D vector, got {len(emb)}"
        print("   >>> AI EMBEDDING EXTRACTION FROM ZEBRONICS FRAME SUCCEEDED!")

    # 4. Error Handling & Disconnection Resilience
    print(f"\n4. Testing Disconnected / Invalid Camera (Index 99):")
    cam_invalid = ZebronicsUSBWebcamSource(camera_index=99)
    inv_ok, inv_frame, inv_meta = cam_invalid.get_frame()
    cam_invalid.release()

    print(f"   Invalid Index Success: {inv_ok}")
    print(f"   Reported Error: {inv_meta.get('error')}")
    assert not inv_ok, "Expected failure for index 99"
    assert inv_meta.get("error") == "ZEBRONICS_CAMERA_DISCONNECTED", f"Unexpected error: {inv_meta.get('error')}"
    print("   >>> DISCONNECTION / INVALID CAMERA ERROR HANDLED GRACEFULLY WITHOUT CRASHING!")

    print("\n==================================================")
    print(" ALL ZEBRONICS CAMERA INTEGRATION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    test_zebronics_full_integration()
