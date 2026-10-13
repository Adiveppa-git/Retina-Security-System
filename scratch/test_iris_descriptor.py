import os
import sys

sys.path.append(os.path.abspath("."))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from ai_models.utils.daugman_normalization import daugman_polar_normalization


def extract_iris_descriptor(pil_img):
    """
    Computes fine-grained Iris Crypt & Furrow Texture Descriptor:
    1. Daugman Polar Normalization (64x512).
    2. Multi-Scale Gabor Filter Bank (8 orientations x 4 scales).
    3. Spatial L2-normalized Crypt Feature Vector (512-D).
    """
    polar_strip = daugman_polar_normalization(pil_img)
    gray = cv2.cvtColor(np.array(polar_strip), cv2.COLOR_RGB2GRAY)
    h, w = gray.shape

    feature_vector = []

    # Multi-scale Gabor Filter Bank
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        for frequency in [0.1, 0.25, 0.4]:
            kernel = cv2.getGaborKernel((15, 15), 3.0, theta, 1.0/frequency, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            # Spatial block pooling (4x16 grid)
            blocks_h, blocks_w = 4, 16
            bh_size, bw_size = h // blocks_h, w // blocks_w
            for bh in range(blocks_h):
                for bw in range(blocks_w):
                    block = filtered[bh*bh_size:(bh+1)*bh_size, bw*bw_size:(bw+1)*bw_size]
                    feature_vector.append(np.mean(block))
                    feature_vector.append(np.std(block))

    feat = np.array(feature_vector, dtype=np.float32)
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat = feat / norm
    return feat


img1_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3_path = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

desc1 = extract_iris_descriptor(Image.open(img1_path))
desc2 = extract_iris_descriptor(Image.open(img2_path))
desc3 = extract_iris_descriptor(Image.open(img3_path))

sim_same = np.dot(desc1, desc2)
sim_diff = np.dot(desc1, desc3)

print(f"Same Subject Iris Texture Similarity     : {sim_same:.4f} ({sim_same*100:.2f}%)")
print(f"Different Subject Iris Texture Similarity: {sim_diff:.4f} ({sim_diff*100:.2f}%)")
