import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ai_models.iris_recognition.model import IrisRecognitionCNN
from ai_models.utils.daugman_normalization import daugman_polar_normalization

device = torch.device("cpu")
metric_model = IrisRecognitionCNN(embedding_dim=512).to(device)
metric_model.eval()

tr = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_512d(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    polar_strip = daugman_polar_normalization(pil_img)
    t = tr(polar_strip).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = metric_model(t)
    return emb


img1 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3 = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

emb1 = extract_512d(img1)
emb2 = extract_512d(img2)
emb3 = extract_512d(img3)

same_sim = F.cosine_similarity(emb1, emb2).item()
diff_sim = F.cosine_similarity(emb1, emb3).item()

print(f"Same Subject 512D Cosine Similarity     : {same_sim:.4f} ({same_sim*100:.2f}%)")
print(f"Different Subject 512D Cosine Similarity: {diff_sim:.4f} ({diff_sim*100:.2f}%)")
