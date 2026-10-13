import os
import sys
import cv2
import glob
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from app.services.iris_detection_service import detect_and_crop_iris

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset/rgb_webcam_dataset"))

def prepare_rgb_dataset():
    print("==================================================")
    print(" PREPARING RGB WEBCAM EYE CROPS DATASET")
    print("==================================================")

    os.makedirs(os.path.join(DATASET_ROOT, "train"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_ROOT, "val"), exist_ok=True)

    # Define User Source Image Maps
    # User 0: Renkigouda (#70)
    # User 1: Adiveppa (#71)
    # User 2: Unknown / Extra User (#99)
    user_sources = {
        0: {
            "name": "Renkigouda",
            "train_sources": ["project_images/002.jpeg", "project_images/020.jpeg"],
            "val_sources": ["project_images/020.jpeg"]
        },
        1: {
            "name": "Adiveppa",
            "train_sources": ["project_images/008.jpeg", "project_images/018.jpeg"],
            "val_sources": ["project_images/008.jpeg"]
        },
        2: {
            "name": "Unknown_Extra",
            "train_sources": ["project_images/012.jpeg", "project_images/016.jpeg"],
            "val_sources": ["project_images/016.jpeg"]
        }
    }

    stats = {
        "total_users": len(user_sources),
        "user_samples": {},
        "left_count": 0,
        "right_count": 0,
        "train_count": 0,
        "val_count": 0,
        "resolutions": [],
        "iris_diameters": []
    }

    def process_and_save(img_path, target_dir, label_idx, user_name, session_name):
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found, skipping.")
            return 0

        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = detect_and_crop_iris(img_rgb)
        payload = res.payload

        count = 0
        h, w, c = img_bgr.shape
        stats["resolutions"].append((w, h))

        if payload.get("left_eye_bbox") and payload.get("left_iris_detected"):
            lx1, ly1, lx2, ly2 = payload["left_eye_bbox"]
            left_crop = img_rgb[ly1:ly2, lx1:lx2]
            if left_crop.size > 0:
                l_r = payload.get("left_iris_radius", 0)
                stats["iris_diameters"].append(2 * l_r)
                stats["left_count"] += 1
                out_path = os.path.join(target_dir, f"u{label_idx}_{user_name}_L_{session_name}_{count}.png")
                Image.fromarray(left_crop).save(out_path)
                count += 1

        if payload.get("right_eye_bbox") and payload.get("right_iris_detected"):
            rx1, ry1, rx2, ry2 = payload["right_eye_bbox"]
            right_crop = img_rgb[ry1:ry2, rx1:rx2]
            if right_crop.size > 0:
                r_r = payload.get("right_iris_radius", 0)
                stats["iris_diameters"].append(2 * r_r)
                stats["right_count"] += 1
                out_path = os.path.join(target_dir, f"u{label_idx}_{user_name}_R_{session_name}_{count}.png")
                Image.fromarray(right_crop).save(out_path)
                count += 1

        return count

    for u_idx, u_info in user_sources.items():
        uname = u_info["name"]
        stats["user_samples"][uname] = 0

        # Process Train Sources
        train_saved = 0
        for s_idx, src in enumerate(u_info["train_sources"]):
            saved = process_and_save(src, os.path.join(DATASET_ROOT, "train"), u_idx, uname, f"sess{s_idx+1}")
            train_saved += saved

        # Process Val Sources
        val_saved = 0
        for s_idx, src in enumerate(u_info["val_sources"]):
            saved = process_and_save(src, os.path.join(DATASET_ROOT, "val"), u_idx, uname, f"val_sess{s_idx+1}")
            val_saved += saved

        stats["train_count"] += train_saved
        stats["val_count"] += val_saved
        stats["user_samples"][uname] = train_saved + val_saved

    print("\n--------------------------------------------------")
    print(" PRE-TRAINING DATASET STATISTICS REPORT")
    print("--------------------------------------------------")
    print(f" 1. Total Number of Enrolled Users : {stats['total_users']}")
    for uname, cnt in stats["user_samples"].items():
        print(f"    - User '{uname:<15}': {cnt} total eye samples")
    print(f" 2. Sample Counts by Eye Side      : Left={stats['left_count']} | Right={stats['right_count']}")
    avg_w = np.mean([r[0] for r in stats["resolutions"]])
    avg_h = np.mean([r[1] for r in stats["resolutions"]])
    print(f" 3. Camera Image Resolution Range  : Mean = {avg_w:.0f}x{avg_h:.0f} (Range {min([r[0] for r in stats['resolutions']])}x{min([r[1] for r in stats['resolutions']])} to {max([r[0] for r in stats['resolutions']])}x{max([r[1] for r in stats['resolutions']])})")
    avg_dia = np.mean(stats["iris_diameters"]) if stats["iris_diameters"] else 0.0
    print(f" 4. Estimated Iris Pixel Diameter   : Mean = {avg_dia:.1f} px (Range {min(stats['iris_diameters']):.1f} - {max(stats['iris_diameters']):.1f} px)")
    print(f" 5. Dataset Split Counts           : Train={stats['train_count']} samples | Val={stats['val_count']} samples | Total={stats['train_count']+stats['val_count']} samples")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    prepare_rgb_dataset()
