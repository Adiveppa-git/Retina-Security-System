import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.database.connection import SessionLocal
from app.services.iris_detection_service import detect_and_crop_iris
from app.services.anti_spoof_service import check_spoof
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding
from app.services.iris_recognition_service import recognize_iris_v2_1to1_diagnostic, recognize_iris_v2_diagnostic

def benchmark_pipeline_latency(num_trials=100):
    print("==================================================")
    print(f" MEASURING ACCESS DECISION LATENCY ({num_trials} TRIALS)")
    print("==================================================")
    
    # Pick a real project image with face and eyes
    test_img_path = "project_images/001.jpeg"
    if not os.path.exists(test_img_path):
        test_img_path = "scratch/ref008_L.jpg"
    
    img = Image.open(test_img_path).convert("RGB")
    img_np = np.array(img)
    
    # Warmup
    print("Warming up models...")
    for _ in range(5):
        _ = detect_and_crop_iris(img_np)
        _ = check_spoof(img)
        _ = extract_v2_embedding(img)
    
    stage1_times = [] # Face & Eye Landmark Detection
    stage2_times = [] # Anti-spoofing Liveness
    stage3_times = [] # Iris Feature Embedding (512-D)
    stage4_times = [] # Database Search / Match Evaluation
    total_pipeline_times = [] # End-to-end access decision

    for i in range(num_trials):
        t0 = time.perf_counter()
        
        # Stage 1: Face & Eye Landmark Localization
        t_s1_start = time.perf_counter()
        det_res = detect_and_crop_iris(img_np)
        t_s1_end = time.perf_counter()
        
        # Stage 2: Anti-Spoofing
        t_s2_start = time.perf_counter()
        _ = check_spoof(img)
        t_s2_end = time.perf_counter()
        
        # Stage 3: Feature Extraction
        t_s3_start = time.perf_counter()
        _ = extract_v2_embedding(img)
        t_s3_end = time.perf_counter()
        
        # Total Full Pipeline Call (including DB query and decision logic)
        t_full_start = time.perf_counter()
        res = recognize_iris_v2_diagnostic(img)
        t_full_end = time.perf_counter()
        
        stage1_times.append((t_s1_end - t_s1_start) * 1000.0)
        stage2_times.append((t_s2_end - t_s2_start) * 1000.0)
        stage3_times.append((t_s3_end - t_s3_start) * 1000.0)
        total_pipeline_times.append((t_full_end - t_full_start) * 1000.0)

    s1 = np.array(stage1_times)
    s2 = np.array(stage2_times)
    s3 = np.array(stage3_times)
    tot = np.array(total_pipeline_times)

    print("\n--------------------------------------------------")
    print(" BREAKDOWN OF DECISION LATENCY PER STAGE (ms):")
    print("--------------------------------------------------")
    print(f"1. Face & Eye Landmark Localization: Mean = {s1.mean():.2f} ms | Median = {np.median(s1):.2f} ms | P95 = {np.percentile(s1, 95):.2f} ms")
    print(f"2. Anti-Spoofing Liveness Inference : Mean = {s2.mean():.2f} ms | Median = {np.median(s2):.2f} ms | P95 = {np.percentile(s2, 95):.2f} ms")
    print(f"3. 512-D Feature Extraction (ArcFace): Mean = {s3.mean():.2f} ms | Median = {np.median(s3):.2f} ms | P95 = {np.percentile(s3, 95):.2f} ms")
    print("--------------------------------------------------")
    print(" TOTAL END-TO-END ACCESS VERIFICATION LATENCY:")
    print("--------------------------------------------------")
    print(f"  Number of Timed Trials : {num_trials}")
    print(f"  Mean Latency           : {tot.mean():.2f} ms")
    print(f"  Median Latency         : {np.median(tot):.2f} ms")
    print(f"  Standard Deviation     : {tot.std():.2f} ms")
    print(f"  Minimum Latency        : {tot.min():.2f} ms")
    print(f"  Maximum Latency        : {tot.max():.2f} ms")
    print(f"  95th Percentile (P95)  : {np.percentile(tot, 95):.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    benchmark_pipeline_latency(100)
