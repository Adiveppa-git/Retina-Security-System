import os
import shutil

real_src = "datasets/anti_spoof_dataset/real"
fake_src = "datasets/anti_spoof_dataset/fake"

real_dst = "datasets/anti_spoof_dataset/real_flat"
fake_dst = "datasets/anti_spoof_dataset/fake_flat"

os.makedirs(real_dst, exist_ok=True)
os.makedirs(fake_dst, exist_ok=True)

def collect_images(src, dst):
    count = 0

    for root, dirs, files in os.walk(src):
        for file in files:

            # accept any image extension
            if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".tif",".tiff")):

                src_path = os.path.join(root, file)

                dst_path = os.path.join(dst, f"{count}_{file}")

                shutil.copy(src_path, dst_path)

                count += 1

    print(f"{count} images copied from {src}")


print("Processing REAL images...")
collect_images(real_src, real_dst)

print("Processing FAKE images...")
collect_images(fake_src, fake_dst)

print("✅ Dataset prepared successfully")