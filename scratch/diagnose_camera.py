import cv2
import time
import numpy as np

def test_camera_capture(index=1, backend=cv2.CAP_MSMF, backend_name="MSMF", warmup_frames=15, width=None, height=None, fourcc=None):
    print(f"\n==================================================")
    print(f"Testing Index {index} with {backend_name}")
    print(f"Requested: width={width}, height={height}, fourcc={fourcc}, warmup={warmup_frames}")
    print(f"==================================================")
    
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        print(f"FAILED to open camera {index} with {backend_name}")
        return None

    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    auto_exp = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
    contrast = cap.get(cv2.CAP_PROP_CONTRAST)
    gain = cap.get(cv2.CAP_PROP_GAIN)
    
    print(f"Actual Config: {actual_w}x{actual_h} @ {actual_fps}fps")
    print(f"Auto-Exposure: {auto_exp}, Exposure: {exp}, Brightness: {brightness}, Contrast: {contrast}, Gain: {gain}")

    # Read frame 0 (immediate)
    ret0, frame0 = cap.read()
    if ret0 and frame0 is not None:
        mean0 = float(np.mean(frame0))
        std0 = float(np.std(frame0))
        print(f"Frame 0 (immediate): Mean brightness = {mean0:.2f}, Std = {std0:.2f}")
        cv2.imwrite(f"scratch/{backend_name}_frame0.jpg", frame0)
    else:
        print("Frame 0: Failed to read")

    # Now read warmup frames to let AE / AGC converge
    t0 = time.time()
    last_frame = None
    means = []
    for i in range(1, warmup_frames + 1):
        ret, frame = cap.read()
        if ret and frame is not None:
            m = float(np.mean(frame))
            means.append(m)
            last_frame = frame
        time.sleep(0.03)
    elapsed = time.time() - t0

    if last_frame is not None:
        last_mean = float(np.mean(last_frame))
        last_std = float(np.std(last_frame))
        print(f"Frame {warmup_frames} (after {elapsed:.2f}s warmup): Mean brightness = {last_mean:.2f}, Std = {last_std:.2f}")
        print(f"Brightness trajectory over warmup: {[round(m, 1) for m in means[:5]]} ... {[round(m, 1) for m in means[-3:]]}")
        filename = f"scratch/{backend_name}_w{int(actual_w)}_warmup.jpg"
        cv2.imwrite(filename, last_frame)
        print(f"Saved: {filename}")
    
    cap.release()
    return last_frame

if __name__ == "__main__":
    # Test 1: DirectShow with default resolution + warmup
    test_camera_capture(1, cv2.CAP_DSHOW, "DSHOW_default", warmup_frames=15)
    
    # Test 2: MSMF with default resolution + warmup
    test_camera_capture(1, cv2.CAP_MSMF, "MSMF_default", warmup_frames=15)

    # Test 3: DirectShow with 1280x720 + warmup
    test_camera_capture(1, cv2.CAP_DSHOW, "DSHOW_720p", warmup_frames=15, width=1280, height=720)

    # Test 4: MSMF with 1280x720 + warmup
    test_camera_capture(1, cv2.CAP_MSMF, "MSMF_720p", warmup_frames=15, width=1280, height=720)

    # Test 5: MSMF with 1920x1080 + warmup
    test_camera_capture(1, cv2.CAP_MSMF, "MSMF_1080p", warmup_frames=15, width=1920, height=1080)
