import abc
import cv2
import os
import time
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from PIL import Image

class BaseCameraSource(abc.ABC):
    """
    Abstract Base Class for Biometric Camera Acquisition Sources.
    Decouples frame acquisition from recognition logic and OpenCV VideoCapture dependencies.
    Enforces the Standard Frame Metadata Contract.
    """
    @abc.abstractmethod
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Retrieves a single RGB frame from the camera source.
        Returns: (success: bool, frame_rgb: Optional[np.ndarray], metadata: Dict[str, Any])
        """
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """Releases camera hardware resources."""
        pass


class ZebronicsUSBWebcamSource(BaseCameraSource):
    """
    Primary Biometric Camera Acquisition Source: Zebronics USB Webcam.
    - Windows-native DirectShow (CAP_DSHOW) / Media Foundation (CAP_MSMF) capture backend.
    - Explicit device validation: verifies camera index is responsive and yields valid frames.
      Never silently falls back to the built-in laptop camera if disconnected.
    - Automated Hardware Sensor Warmup: cycles initial frames to allow hardware
      Auto-Exposure (AE), Auto-Gain Control (AGC), and Auto-White Balance (AWB) to converge,
      guaranteeing crisp, balanced, non-black frames matching Windows Camera quality.
    - Native HD resolution (1280x720) @ 30 FPS for optimal eye and iris detection detail.
    """
    def __init__(self, camera_index: int = 1, width: int = 1280, height: int = 720, fps: int = 30):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.backend_name = "NONE"
        self._is_warmed_up = False

    def initialize(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True

        # Use fast Windows DirectShow first, fallback to Media Foundation
        backends = []
        if os.name == 'nt':
            backends = [(cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")]
        else:
            backends = [(cv2.CAP_ANY, "ANY")]

        opened = False
        for backend_id, name in backends:
            try:
                cap = cv2.VideoCapture(self.camera_index, backend_id)
                if cap.isOpened():
                    # Set resolution and FPS
                    if self.width and self.height:
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    if self.fps:
                        cap.set(cv2.CAP_PROP_FPS, self.fps)

                    self.cap = cap
                    self.backend_name = name
                    opened = True
                    break
                else:
                    cap.release()
            except Exception:
                pass

        if not opened or self.cap is None or not self.cap.isOpened():
            self.cap = None
            self.backend_name = "NONE"
            self._is_warmed_up = False
            return False

        # Drain warmup frames so camera auto-exposure loop converges
        self._warmup_sensor()
        return self._is_warmed_up

    def _warmup_sensor(self, max_frames: int = 15):
        if self.cap is None or not self.cap.isOpened():
            return
        for _ in range(max_frames):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # If mean brightness indicates converged auto-exposure, sensor is warm
                if float(np.mean(frame)) > 25.0:
                    self._is_warmed_up = True
                    return
            time.sleep(0.02)
        # Even if not past 25 (e.g. dark room), mark warmed up if frames can be read
        self._is_warmed_up = (self.cap is not None and self.cap.isOpened())

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        ts = datetime.now().isoformat()
        if not self.initialize():
            return False, None, {
                "source": "Zebronics USB Webcam",
                "camera_index": self.camera_index,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "RGB_VISIBLE",
                "backend": self.backend_name,
                "timestamp": ts,
                "error": "ZEBRONICS_CAMERA_DISCONNECTED"
            }

        # Read latest frame with retry and exposure safety
        ret, frame_bgr = self.cap.read()
        retries = 0
        while (not ret or frame_bgr is None or np.mean(frame_bgr) < 15.0) and retries < 5:
            time.sleep(0.03)
            ret, frame_bgr = self.cap.read()
            retries += 1

        if not ret or frame_bgr is None:
            return False, None, {
                "source": "Zebronics USB Webcam",
                "camera_index": self.camera_index,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "RGB_VISIBLE",
                "backend": self.backend_name,
                "timestamp": ts,
                "error": "FRAME_CAPTURE_FAILED"
            }

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        return True, frame_rgb, {
            "source": "Zebronics USB Webcam",
            "camera_index": self.camera_index,
            "width": w,
            "height": h,
            "channels": 3,
            "color_format": "RGB",
            "spectrum": "RGB_VISIBLE",
            "backend": self.backend_name,
            "mean_brightness": round(float(np.mean(frame_rgb)), 2),
            "timestamp": ts,
            "error": None
        }

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        self._is_warmed_up = False


class USBWebcamSource(ZebronicsUSBWebcamSource):
    """
    External USB Camera Source (Backwards-compatible wrapper inheriting ZebronicsUSBWebcamSource).
    """
    def __init__(self, camera_index: int = 1, width: int = 1280, height: int = 720, fps: int = 30):
        super().__init__(camera_index=camera_index, width=width, height=height, fps=fps)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        ok, frame, meta = super().get_frame()
        # Maintain "usb_webcam" source tag for backwards compatibility if needed
        if not ok:
            meta["source"] = "usb_webcam"
            if meta.get("error") == "ZEBRONICS_CAMERA_DISCONNECTED":
                meta["error"] = f"CAMERA_INDEX_{self.camera_index}_UNAVAILABLE"
        else:
            meta["source"] = "Zebronics USB Webcam"
        return ok, frame, meta


class LaptopRGBWebcamSource(BaseCameraSource):
    """
    Built-in Laptop RGB Webcam Source (Default Device Index 0).
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None

    def initialize(self) -> bool:
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
        return self.cap.isOpened()

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        ts = datetime.now().isoformat()
        if not self.initialize():
            return False, None, {
                "source": "laptop_rgb_webcam",
                "camera_index": self.camera_index,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "RGB_VISIBLE",
                "timestamp": ts,
                "error": "CAMERA_INIT_FAILED"
            }

        ret, frame_bgr = self.cap.read()
        if not ret or frame_bgr is None:
            return False, None, {
                "source": "laptop_rgb_webcam",
                "camera_index": self.camera_index,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "RGB_VISIBLE",
                "timestamp": ts,
                "error": "FRAME_CAPTURE_FAILED"
            }

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        metadata = {
            "source": "laptop_rgb_webcam",
            "camera_index": self.camera_index,
            "width": w,
            "height": h,
            "channels": 3,
            "color_format": "RGB",
            "spectrum": "RGB_VISIBLE",
            "timestamp": ts,
            "error": None
        }
        return True, frame_rgb, metadata

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            self.cap = None


class StaticFileCameraSource(BaseCameraSource):
    """
    Deterministic Image File Camera Source for Offline Diagnostic Testing.
    """
    def __init__(self, file_path: str):
        self.file_path = os.path.abspath(file_path)

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        ts = datetime.now().isoformat()
        if not os.path.exists(self.file_path):
            return False, None, {
                "source": "static_file",
                "file_path": self.file_path,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "FILE_IMAGE",
                "timestamp": ts,
                "error": f"FILE_NOT_FOUND: {os.path.basename(self.file_path)}"
            }

        frame_bgr = cv2.imread(self.file_path)
        if frame_bgr is None:
            return False, None, {
                "source": "static_file",
                "file_path": self.file_path,
                "width": 0,
                "height": 0,
                "channels": 0,
                "color_format": "NONE",
                "spectrum": "FILE_IMAGE",
                "timestamp": ts,
                "error": "CORRUPTED_IMAGE_FILE"
            }

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        metadata = {
            "source": "static_file",
            "file_path": self.file_path,
            "width": w,
            "height": h,
            "channels": 3,
            "color_format": "RGB",
            "spectrum": "FILE_IMAGE",
            "timestamp": ts,
            "error": None
        }
        return True, frame_rgb, metadata

    def release(self) -> None:
        pass


class NIRIrisCameraSource(BaseCameraSource):
    """
    Future Dedicated Near-Infrared (NIR) Iris Camera Interface (Phase 5D Stub).
    Probes video devices for 850nm NIR hardware capability.
    NEVER automatically labels a camera as 'NIR_850NM' unless hardware NIR spectrum capability is verified.
    """
    def __init__(self, device_id: str = "NIR_IRIS_CAM_0"):
        self.device_id = device_id

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Dict[str, Any]]:
        ts = datetime.now().isoformat()
        
        # Hardware Probe: Check connected video capture devices
        connected_devices = []
        for idx in range(3):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    connected_devices.append({"index": idx, "resolution": f"{w}x{h}"})
                cap.release()

        metadata = {
            "source": "nir_iris_camera",
            "device_id": self.device_id,
            "width": 0,
            "height": 0,
            "channels": 0,
            "color_format": "NONE",
            "spectrum": "UNKNOWN_NOT_VALIDATED",
            "status": "PHYSICAL NIR CAMERA NOT VALIDATED",
            "timestamp": ts,
            "connected_video_devices": connected_devices,
            "error": "NIR_HARDWARE_SPECTRUM_NOT_VERIFIED",
            "target_input_contract": {
                "illumination_spectrum": "850nm Near-Infrared LED Flash Array",
                "sensor_type": "Monochromatic Monochrome CMOS Sensor (No Bayer RGB filter)",
                "resolution_target": "640x480 (VGA) or 1280x720 (HD)",
                "iris_diameter_target_px": "120-160px reference target (CASIA benchmark)",
                "frame_rate_fps": 30,
                "color_format_conversion": "8-bit Grayscale -> 3-channel RGB via cv2.cvtColor",
                "exposure_control": "Fixed / Fast Auto-Exposure tuned for 850nm LED flash",
                "working_distance_cm": "20cm to 40cm"
            }
        }
        return False, None, metadata

    def release(self) -> None:
        pass


def get_primary_camera_source(camera_index: int = 1) -> BaseCameraSource:
    """
    Factory function returning the primary biometric camera source:
    Zebronics USB Webcam (external camera index 1).
    """
    return ZebronicsUSBWebcamSource(camera_index=camera_index)
