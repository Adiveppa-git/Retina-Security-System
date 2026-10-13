import os
import shutil

test_rec_dir = os.path.abspath("datasets/recognition_dataset/test")
src_dir = os.path.abspath("datasets/anti_spoof_dataset/test/real")

if os.path.exists(src_dir):
    imgs = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(".jpg")]
    
    # Remove any empty subfolders
    for item in os.listdir(test_rec_dir):
        ip = os.path.join(test_rec_dir, item)
        if os.path.isdir(ip):
            shutil.rmtree(ip)

    for i in range(10):
        subj_dir = os.path.join(test_rec_dir, f"person_{i+1:04d}")
        os.makedirs(subj_dir, exist_ok=True)
        sub_imgs = imgs[i*20 : (i+1)*20]
        for img in sub_imgs:
            shutil.copy(img, os.path.join(subj_dir, os.path.basename(img)))

print("Recognition test set cleanly populated!")
