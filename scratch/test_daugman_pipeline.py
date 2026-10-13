import os
import sys

sys.path.append(os.path.abspath("."))

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN
from ai_models.utils.daugman_normalization import daugman_polar_normalization

# Load recognition model
device = torch.device("cpu")
model_path = "ai_models/trained_models/iris_feature_model.pth"
m = IrisIdentityCNN(num_classes=2).to(device)
m.load_state_dict(torch.load(model_path, map_location=device))
m.eval()

tr = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_embedding(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    # Apply Daugman Polar Normalization
    polar_strip = daugman_polar_normalization(pil_img)
    tensor_img = tr(polar_strip).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = m.features(tensor_img)
        feat = m.avgpool(feat)
        feat = torch.flatten(feat, 1)
        emb = m.classifier[:2](feat)
        emb = F.normalize(emb, p=2, dim=1)
    return emb


img1 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3 = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

emb1 = get_embedding(img1)
emb2 = get_embedding(img2)
emb3 = get_embedding(img3)

same_sim = F.cosine_similarity(emb1, emb2).item()
diff_sim = F.cosine_similarity(emb1, emb3).item()

print(f"Same Subject Daugman Polar Similarity: {same_sim:.4f} ({same_sim*100:.2f}%)")
print(f"Different Subject Daugman Polar Similarity: {diff_sim:.4f} ({diff_sim*100:.2f}%)")
