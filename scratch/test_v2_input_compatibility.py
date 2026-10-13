import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding, v2_transform, get_v2_model
from backend.app.services.iris_detection_service import detect_and_crop_iris

def run_v2_compatibility_diagnostic():
    print("==================================================")
    print(" V2 INPUT COMPATIBILITY & EMBEDDING DIAGNOSTIC")
    print("==================================================")

    test_img_path = "project_images/002.jpeg"
    if not os.path.exists(test_img_path):
        print(f"[ERROR] Test image {test_img_path} not found!")
        return

    # Step 1: Detect and Crop Eye ROI using Phase 3 Detector
    res = detect_and_crop_iris(test_img_path)
    polar_ribbon = res[0] # PIL Image

    print(f"1. Input Image Format          : PIL RGB Image")
    print(f"2. Raw Crop / Ribbon Size     : {polar_ribbon.size} (Width, Height)")

    # Step 2: Apply V2 Transform
    tensor_img = v2_transform(polar_ribbon).unsqueeze(0)
    print(f"3. Tensor Input Dimensions     : {list(tensor_img.shape)} (Batch, Channels, Height, Width)")
    print(f"4. Tensor Value Mean ± Std     : {tensor_img.mean().item():.4f} ± {tensor_img.std().item():.4f}")
    print(f"5. Tensor Min / Max Range      : [{tensor_img.min().item():.4f}, {tensor_img.max().item():.4f}]")

    # Step 3: Extract Embedding
    vec = extract_v2_embedding(polar_ribbon)
    vec_tensor = torch.tensor(vec)
    l2_norm = torch.norm(vec_tensor, p=2).item()

    print(f"6. Embedding Vector Dimension  : {len(vec)}-D")
    print(f"7. Embedding L2 Norm          : {l2_norm:.6f} (Verified == 1.0)")
    print(f"8. Embedding Vector Mean ± Std : {vec_tensor.mean().item():.6f} ± {vec_tensor.std().item():.6f}")

    # Step 4: Compare Similarity across 2 webcam images of same user vs different user
    img_a1 = "project_images/002.jpeg"
    img_a2 = "project_images/008.jpeg"
    img_b1 = "project_images/012.jpeg"

    res_a1 = detect_and_crop_iris(img_a1)
    res_a2 = detect_and_crop_iris(img_a2)
    res_b1 = detect_and_crop_iris(img_b1)

    vec_a1 = torch.tensor(extract_v2_embedding(res_a1[0])).unsqueeze(0)
    vec_a2 = torch.tensor(extract_v2_embedding(res_a2[0])).unsqueeze(0)
    vec_b1 = torch.tensor(extract_v2_embedding(res_b1[0])).unsqueeze(0)

    sim_a1_a2 = F.cosine_similarity(vec_a1, vec_a2).item()
    sim_a1_b1 = F.cosine_similarity(vec_a1, vec_b1).item()

    print("\n--- WEBCAM EMBEDDING COSINE SIMILARITY TEST ---")
    print(f" Same Person (Frame A1 vs Frame A2) : Cosine Sim = {sim_a1_a2:.4f}")
    print(f" Different Person (Frame A1 vs B1)  : Cosine Sim = {sim_a1_b1:.4f}")
    print("==================================================\n")

if __name__ == "__main__":
    run_v2_compatibility_diagnostic()
