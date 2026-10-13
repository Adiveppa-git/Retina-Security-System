import os
import sys

sys.path.append(os.path.abspath("."))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN

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


def get_raw_feature(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    t = tr(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = m.features(t)
        feat = m.avgpool(feat)
        feat = torch.flatten(feat, 1)[0]
    return feat


img1_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2_path = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3_path = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

f1 = get_raw_feature(img1_path)
f2 = get_raw_feature(img2_path)
f3 = get_raw_feature(img3_path)

# Mean vector of common MobileNet activations
mean_feat = (f1 + f2 + f3) / 3.0

c1 = F.normalize(f1 - mean_feat, p=2, dim=0)
c2 = F.normalize(f2 - mean_feat, p=2, dim=0)
c3 = F.normalize(f3 - mean_feat, p=2, dim=0)

sim_same = torch.dot(c1, c2).item()
sim_diff = torch.dot(c1, c3).item()

print(f"Mean-Centered Same Subject Similarity     : {sim_same:.4f} ({sim_same*100:.2f}%)")
print(f"Mean-Centered Different Subject Similarity: {sim_diff:.4f} ({sim_diff*100:.2f}%)")
