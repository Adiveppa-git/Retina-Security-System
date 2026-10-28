import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

from app.services.iris_detection_service import detect_and_crop_iris
from app.services.anti_spoof_service import check_spoof
from ai_models.iris_recognition.v2_embedding_pipeline import extract_v2_embedding
from app.services.iris_recognition_service import recognize_iris_v2_diagnostic

def benchmark_true_pipeline(num_trials=50):
    img_path = "project_images/008.jpeg"
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # Warmup
    for _ in range(5):
        _ = recognize_iris_v2_diagnostic(img)
        _ = check_spoof(img)

    stage_det_times = []
    stage_as_times = []
    stage_emb_left_times = []
    stage_emb_right_times = []
    total_diag_pipeline_times = [] # The actual recognize_iris_v2_diagnostic call
    total_full_with_as_times = [] # Full pipeline IF anti-spoofing is included

    for _ in range(num_trials):
        # 1. Test recognize_iris_v2_diagnostic directly
        t0 = time.perf_counter()
        res = recognize_iris_v2_diagnostic(img)
        t_diag = (time.perf_counter() - t0) * 1000.0
        total_diag_pipeline_times.append(t_diag)

        # 2. Stage Breakdown
        t_det0 = time.perf_counter()
        det_res = detect_and_crop_iris(img_np)
        t_det = (time.perf_counter() - t_det0) * 1000.0
        stage_det_times.append(t_det)

        lx1, ly1, lx2, ly2 = det_res.payload["left_eye_bbox"]
        rx1, ry1, rx2, ry2 = det_res.payload["right_eye_bbox"]
        left_crop = Image.fromarray(img_np[ly1:ly2, lx1:lx2])
        right_crop = Image.fromarray(img_np[ry1:ry2, rx1:rx2])

        # Anti-spoofing on eye crop
        t_as0 = time.perf_counter()
        _ = check_spoof(left_crop)
        t_as = (time.perf_counter() - t_as0) * 1000.0
        stage_as_times.append(t_as)

        # Left embedding
        t_el0 = time.perf_counter()
        _ = extract_v2_embedding(left_crop)
        t_el = (time.perf_counter() - t_el0) * 1000.0
        stage_emb_left_times.append(t_el)

        # Right embedding
        t_er0 = time.perf_counter()
        _ = extract_v2_embedding(right_crop)
        t_er = (time.perf_counter() - t_er0) * 1000.0
        stage_emb_right_times.append(t_er)

        total_full_with_as_times.append(t_diag + t_as)

    print("==================================================")
    print(f" BENCHMARK OF TRUE VERIFICATION PIPELINE ({num_trials} TRIALS)")
    print("==================================================")
    print(f"Test Image: {img_path} (Valid face, dual eyes & dual irises detected)")
    print("--------------------------------------------------")
    print(f"Stage 1: MediaPipe Face & Eye Landmark Detection:")
    print(f"  Mean: {np.mean(stage_det_times):.2f} ms | Median: {np.median(stage_det_times):.2f} ms | P95: {np.percentile(stage_det_times, 95):.2f} ms")
    print(f"Stage 2: Anti-Spoofing Liveness Inference (on cropped eye):")
    print(f"  Mean: {np.mean(stage_as_times):.2f} ms | Median: {np.median(stage_as_times):.2f} ms | P95: {np.percentile(stage_as_times, 95):.2f} ms")
    print(f"Stage 3a: Left Eye 512-D ArcFace Feature Extraction:")
    print(f"  Mean: {np.mean(stage_emb_left_times):.2f} ms | Median: {np.median(stage_emb_left_times):.2f} ms | P95: {np.percentile(stage_emb_left_times, 95):.2f} ms")
    print(f"Stage 3b: Right Eye 512-D ArcFace Feature Extraction:")
    print(f"  Mean: {np.mean(stage_emb_right_times):.2f} ms | Median: {np.median(stage_emb_right_times):.2f} ms | P95: {np.percentile(stage_emb_right_times, 95):.2f} ms")
    print("--------------------------------------------------")
    print(f"Actual recognize_iris_v2_diagnostic Pipeline (Det + Dual Emb + 20 DB Templates Cosine Matching):")
    print(f"  Mean: {np.mean(total_diag_pipeline_times):.2f} ms | Median: {np.median(total_diag_pipeline_times):.2f} ms | P95: {np.percentile(total_diag_pipeline_times, 95):.2f} ms")
    print(f"Theoretical Full Pipeline Including Anti-Spoofing:")
    print(f"  Mean: {np.mean(total_full_with_as_times):.2f} ms | Median: {np.median(total_full_with_as_times):.2f} ms | P95: {np.percentile(total_full_with_as_times, 95):.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    benchmark_true_pipeline(50)
