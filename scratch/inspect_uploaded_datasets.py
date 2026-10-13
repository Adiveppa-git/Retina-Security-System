import os
import cv2
from PIL import Image


def inspect_directory(name, dir_path):
    print(f"\n==================================================")
    print(f" DEEP INSPECTION REPORT: {name}")
    print(f" Path: {dir_path}")
    print(f"==================================================")

    if not os.path.exists(dir_path):
        print(" Directory not found!")
        return

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    all_files = []
    subfolders = set()

    for root, dirs, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(valid_extensions):
                all_files.append(os.path.join(root, f))
                subfolders.add(root)

    print(f"  Total Valid Images Found : {len(all_files)}")
    print(f"  Total Subdirectories     : {len(subfolders)}")

    if not all_files:
        return

    # Sample Inspection (check resolution, channels, corruption)
    valid_count = 0
    corrupted_count = 0
    sample_details = []

    for i, img_path in enumerate(all_files[:15]):  # Inspect first 15 images
        try:
            img = Image.open(img_path)
            w, h = img.size
            mode = img.mode
            valid_count += 1
            sample_details.append(f"   [{i+1}] {os.path.basename(img_path)} | Size: {w}x{h} | Channels/Mode: {mode}")
        except Exception as e:
            corrupted_count += 1

    print(f"  Sample Images Details (Console Output):")
    for detail in sample_details:
        print(detail)

    print(f"--------------------------------------------------")
    print(f"  Corrupted / Missing Data Check: {corrupted_count} corrupted images found.")
    print(f"==================================================\n")


def run_deep_inspection():
    base_downloads = "c:\\Users\\renki\\Downloads"
    targets = [
        ("IITD Iris Database", os.path.join(base_downloads, "IITD_database")),
        ("UBIRIS.v2 Part 1", os.path.join(base_downloads, "ubiris2_1")),
        ("UBIRIS.v2 Part 2", os.path.join(base_downloads, "ubiris2_2")),
        ("Archive (1) Dataset", os.path.join(base_downloads, "archive (1) (1)")),
        ("Archive (2) Dataset", os.path.join(base_downloads, "archive (2)")),
        ("Archive (3) Dataset", os.path.join(base_downloads, "archive (3)")),
        ("Archive (4) Dataset", os.path.join(base_downloads, "archive (4)")),
    ]

    for name, path in targets:
        inspect_directory(name, path)


if __name__ == "__main__":
    run_deep_inspection()
