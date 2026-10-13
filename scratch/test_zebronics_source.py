import cv2
import os
import time
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

def test_source_implementation():
    print("Testing Zebronics camera source prototype...")
    
    # 1. Open with CAP_MSMF
    cap = cv2.VideoCapture(1, cv2.CAP_MSMF if os.name == 'nt' else cv2.CAP_ANY)
    if not cap.isOpened():
        print("Failed to open with MSMF, trying DSHOW...")
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    
    if not cap.isOpened():
        print("Camera index 1 could not be opened!")
        return
    
    # Configure 1280x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Sensor warmup
    warmed = False
    for i in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            mean_val = float(np.mean(frame))
            if mean_val > 25.0:
                print(f"Warmup converged at frame {i+1} with mean brightness {mean_val:.2f}")
                warmed = True
                break
        time.sleep(0.02)

    # Now capture test frame
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print("Capture failed")
        cap.release()
        return

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mean_val = float(np.mean(frame_rgb))
    std_val = float(np.std(frame_rgb))
    h, w = frame_rgb.shape[:2]
    
    print(f"Captured frame: {w}x{h}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")
    assert mean_val > 40.0, f"Image is too dark: mean brightness {mean_val}"
    
    cv2.imwrite("scratch/prototype_zebronics_frame.jpg", frame_bgr)
    print("Frame saved to scratch/prototype_zebronics_frame.jpg")
    cap.release()
    print("Test passed successfully!")

if __name__ == "__main__":
    test_source_implementation()
