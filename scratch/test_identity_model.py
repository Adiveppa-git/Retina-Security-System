import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN
from ai_models.utils.daugman_normalization import daugman_polar_normalization

device = torch.device("cpu")
model_path = "ai_models/trained_models/iris_feature_model.pth"
identity_model = IrisIdentityCNN(num_classes=2).to(device)
identity_model.load_state_dict(torch.load(model_path, map_location=device))
identity_model.eval()

tr = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_logits(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    polar_strip = daugman_polar_normalization(pil_img)
    t = tr(polar_strip).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = identity_model(t)
        probs = F.softmax(logits, dim=1)
    return logits, probs


img1 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3 = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

log1, prob1 = get_logits(img1)
log2, prob2 = get_logits(img2)
log3, prob3 = get_logits(img3)

print("Image 1 (CASIA-Iris-Lamp 0):", prob1.numpy().tolist())
print("Image 2 (CASIA-Iris-Lamp 1):", prob2.numpy().tolist())
print("Image 3 (CASIA-Iris-Thousand 0):", prob3.numpy().tolist())
