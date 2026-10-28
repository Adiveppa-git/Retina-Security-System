import sys, os, numpy as np
from PIL import Image
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))
from app.services.iris_detection_service import detect_and_crop_iris

for f in sorted(os.listdir('project_images')):
    if f.endswith(('.jpeg', '.jpg', '.png')):
        p = os.path.join('project_images', f)
        img = Image.open(p).convert('RGB')
        res = detect_and_crop_iris(np.array(img))
        pl = res.payload
        print(f"{f}: face={pl.get('face_detected')}, L_iris={pl.get('left_iris_detected')}, R_iris={pl.get('right_iris_detected')}")
