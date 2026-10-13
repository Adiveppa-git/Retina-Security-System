import os
import sys

sys.path.append(os.path.abspath("."))

import cv2
import numpy as np
from PIL import Image
from ai_models.utils.daugman_normalization import daugman_polar_normalization


def extract_daugman_iris_code(pil_img):
    """
    Classic Daugman 2D Gabor Phase Quantization (2048-bit Iris Code):
    1. Daugman Polar Normalization (64x512).
    2. Apply 2D Gabor Wavelets (Real & Imaginary components).
    3. Phase quantization to 2-bit binary codes per pixel.
    """
    polar_strip = daugman_polar_normalization(pil_img)
    gray = cv2.cvtColor(np.array(polar_strip), cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Daugman 2D Gabor Wavelet
    wavelength = 8.0
    orientation = 0.0
    sigma_x = 3.0
    sigma_y = 1.5

    # Real Gabor kernel
    gabor_real = cv2.getGaborKernel((15, 15), sigma_x, orientation, wavelength, 0.5, 0, ktype=cv2.CV_32F)
    # Imaginary Gabor kernel (phase shift pi/2)
    gabor_imag = cv2.getGaborKernel((15, 15), sigma_x, orientation, wavelength, 0.5, np.pi / 2, ktype=cv2.CV_32F)

    resp_real = cv2.filter2D(gray, cv2.CV_32F, gabor_real)
    resp_imag = cv2.filter2D(gray, cv2.CV_32F, gabor_imag)

    # 2-bit Phase Quantization
    code1 = (resp_real > 0).astype(np.uint8)
    code2 = (resp_imag > 0).astype(np.uint8)

    # Combine into binary bitvector
    iris_code = np.packbits(np.concatenate([code1.flatten(), code2.flatten()]))
    return iris_code


def compute_daugman_similarity(code_a, code_b):
    """Computes Daugman Biometric Similarity (1.0 - Fractional Hamming Distance)."""
    xor_result = np.bitwise_xor(code_a, code_b)
    # Count set bits
    differing_bits = np.unpackbits(xor_result).sum()
    total_bits = len(code_a) * 8
    hamming_distance = differing_bits / total_bits
    similarity = (1.0 - hamming_distance)
    return similarity


img1_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3_path = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

code1 = extract_daugman_iris_code(Image.open(img1_path))
code2 = extract_daugman_iris_code(Image.open(img2_path))
code3 = extract_daugman_iris_code(Image.open(img3_path))

sim_same = compute_daugman_similarity(code1, code2)
sim_diff = compute_daugman_similarity(code1, code3)

print(f"Daugman Same Subject Iris Code Similarity     : {sim_same:.4f} ({sim_same*100:.2f}%)")
print(f"Daugman Different Subject Iris Code Similarity: {sim_diff:.4f} ({sim_diff*100:.2f}%)")
