import os
import glob
from PIL import Image

def inspect_datasets():
    base_dir = os.path.abspath("datasets")
    
    print("==================================================")
    print("      PURE DATASET STRUCTURE INSPECTION REPORT    ")
    print("==================================================")

    # 1. CASIA-Iris-Lamp
    lamp_dir = os.path.join(base_dir, "CASIA-Iris-Lamp")
    if os.path.exists(lamp_dir):
        lamp_subjs = [d for d in os.listdir(lamp_dir) if os.path.isdir(os.path.join(lamp_dir, d))]
        lamp_img_count = 0
        lamp_l_count = 0
        lamp_r_count = 0
        sample_img_path = None

        for s in lamp_subjs:
            spath = os.path.join(lamp_dir, s)
            for root, _, files in os.walk(spath):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        lamp_img_count += 1
                        if "L" in root or "L" in f:
                            lamp_l_count += 1
                        elif "R" in root or "R" in f:
                            lamp_r_count += 1
                        if not sample_img_path:
                            sample_img_path = os.path.join(root, f)

        img_size = Image.open(sample_img_path).size if sample_img_path else "N/A"
        print(f"\n[1] CASIA-Iris-Lamp:")
        print(f"  - Subject Count     : {len(lamp_subjs)} subjects ({lamp_subjs[0]} to {lamp_subjs[-1]})")
        print(f"  - Total Images      : {lamp_img_count}")
        print(f"  - Left Eye Images   : {lamp_l_count}")
        print(f"  - Right Eye Images  : {lamp_r_count}")
        print(f"  - Images per Subject: ~{(lamp_img_count / max(1, len(lamp_subjs))):.1f} images/subject")
        print(f"  - Resolution        : {img_size} (Width x Height)")
        print(f"  - Image Type        : Cropped NIR Iris / Eye Image")

    # 2. CASIA-Iris-Thousand
    thousand_dir = os.path.join(base_dir, "CASIA-Iris-Thousand")
    if os.path.exists(thousand_dir):
        th_subjs = [d for d in os.listdir(thousand_dir) if os.path.isdir(os.path.join(thousand_dir, d))]
        th_img_count = 0
        th_l_count = 0
        th_r_count = 0
        th_sample = None

        for s in th_subjs:
            spath = os.path.join(thousand_dir, s)
            for root, _, files in os.walk(spath):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        th_img_count += 1
                        if "L" in root or "L" in f:
                            th_l_count += 1
                        elif "R" in root or "R" in f:
                            th_r_count += 1
                        if not th_sample:
                            th_sample = os.path.join(root, f)

        th_size = Image.open(th_sample).size if th_sample else "N/A"
        print(f"\n[2] CASIA-Iris-Thousand:")
        print(f"  - Subject Count     : {len(th_subjs)} subjects ({th_subjs[0]} to {th_subjs[-1]})")
        print(f"  - Total Images      : {th_img_count}")
        print(f"  - Left Eye Images   : {th_l_count}")
        print(f"  - Right Eye Images  : {th_r_count}")
        print(f"  - Images per Subject: ~{(th_img_count / max(1, len(th_subjs))):.1f} images/subject")
        print(f"  - Resolution        : {th_size} (Width x Height)")
        print(f"  - Image Type        : Cropped NIR Iris / Eye Image")

    # 3. IITD Database
    iitd_dir = os.path.join(base_dir, "IITD_database")
    if os.path.exists(iitd_dir):
        iitd_subjs = [d for d in os.listdir(iitd_dir) if os.path.isdir(os.path.join(iitd_dir, d)) and d != "Normalized_Images"]
        iitd_imgs = glob.glob(os.path.join(iitd_dir, "*", "*.bmp"))
        iitd_sample = Image.open(iitd_imgs[0]).size if iitd_imgs else "N/A"
        print(f"\n[3] IIT Delhi Iris Database v1.0:")
        print(f"  - Subject Count     : {len(iitd_subjs)} subjects")
        print(f"  - Total Images      : {len(iitd_imgs)}")
        print(f"  - Resolution        : {iitd_sample}")

    # 4. UBIRIS.v2 (Fake / Spoof / Unconstrained Noise)
    ub1 = os.path.join(base_dir, "ubiris2_1")
    ub2 = os.path.join(base_dir, "ubiris2_2")
    ub_imgs = glob.glob(os.path.join(ub1, "**", "*.tiff"), recursive=True) + glob.glob(os.path.join(ub2, "**", "*.tiff"), recursive=True)
    print(f"\n[4] UBIRIS.v2 (Noisy / At-a-Distance Dataset):")
    print(f"  - Total Images      : {len(ub_imgs)}")

    # 5. Anti-Spoof Dataset Split
    as_train_real = glob.glob(os.path.join(base_dir, "anti_spoof_dataset/train/real", "*"))
    as_train_fake = glob.glob(os.path.join(base_dir, "anti_spoof_dataset/train/fake", "*"))
    as_test_real = glob.glob(os.path.join(base_dir, "anti_spoof_dataset/test/real", "*"))
    as_test_fake = glob.glob(os.path.join(base_dir, "anti_spoof_dataset/test/fake", "*"))

    print(f"\n[5] Anti-Spoofing Dataset Status:")
    print(f"  - Train Real Images : {len(as_train_real):,}")
    print(f"  - Train Fake Images : {len(as_train_fake):,}")
    print(f"  - Test Real Images  : {len(as_test_real):,}")
    print(f"  - Test Fake Images  : {len(as_test_fake):,}")
    print(f"  - Total Real/Fake   : {len(as_train_real)+len(as_test_real):,} Real | {len(as_train_fake)+len(as_test_fake):,} Fake")

if __name__ == "__main__":
    inspect_datasets()
