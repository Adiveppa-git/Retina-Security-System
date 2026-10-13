import cv2
import numpy as np
from PIL import Image


def detect_iris_circles(gray_img):
    """
    Find pupil and iris concentric circles using Hough Circle Transform & Gradients.
    Returns: (pupil_x, pupil_y, pupil_r), (iris_x, iris_y, iris_r)
    """
    # Smooth image to reduce high frequency noise
    blurred = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    h, w = gray_img.shape
    min_dim = min(h, w)
    
    # 1. Detect Pupil (Dark central region)
    pupil_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w // 2,
        param1=50,
        param2=25,
        minRadius=int(min_dim * 0.08),
        maxRadius=int(min_dim * 0.28)
    )

    if pupil_circles is not None:
        pupil_circles = np.uint16(np.round(pupil_circles))
        xp, yp, rp = pupil_circles[0][0]
    else:
        # Fallback estimation to center if circle detection falls below threshold
        xp, yp, rp = w // 2, h // 2, int(min_dim * 0.15)

    # 2. Detect Iris Outer Boundary
    iris_circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=w // 2,
        param1=30,
        param2=20,
        minRadius=int(rp * 1.4),
        maxRadius=int(min_dim * 0.48)
    )

    if iris_circles is not None:
        iris_circles = np.uint16(np.round(iris_circles))
        xi, yi, ri = iris_circles[0][0]
    else:
        xi, yi, ri = xp, yp, int(rp * 2.2)

    return (int(xp), int(yp), int(rp)), (int(xi), int(yi), int(ri))


def daugman_polar_normalization(image, radial_resolution=64, angular_resolution=512):
    """
    Transforms the annular iris ring into a scale and dilation invariant
    rectangular polar coordinate strip (radial_resolution x angular_resolution).
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Detect Pupil and Iris Boundaries
    (xp, yp, rp), (xi, yi, ri) = detect_iris_circles(gray)

    # Angle array theta from 0 to 2*pi
    thetas = np.linspace(0, 2 * np.pi, angular_resolution, endpoint=False)
    r_samples = np.linspace(0, 1, radial_resolution)

    # Polar normalized grid initialization
    polar_strip = np.zeros((radial_resolution, angular_resolution), dtype=np.uint8)

    for i, r in enumerate(r_samples):
        # Linear interpolation between pupil edge and outer iris edge
        xc = (1 - r) * (xp + rp * np.cos(thetas)) + r * (xi + ri * np.cos(thetas))
        yc = (1 - r) * (yp + rp * np.sin(thetas)) + r * (yi + ri * np.sin(thetas))

        # Clamp coordinates to image dimensions
        xc = np.clip(np.round(xc).astype(int), 0, gray.shape[1] - 1)
        yc = np.clip(np.round(yc).astype(int), 0, gray.shape[0] - 1)

        polar_strip[i, :] = gray[yc, xc]

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance iris crypt details
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_strip = clahe.apply(polar_strip)

    # Convert back to 3-channel RGB image for neural network input
    rgb_strip = cv2.cvtColor(enhanced_strip, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(rgb_strip)
