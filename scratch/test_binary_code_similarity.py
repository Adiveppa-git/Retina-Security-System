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


def get_binary_code(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    t = tr(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = m.features(t)
        feat = m.avgpool(feat)
        feat = torch.flatten(feat, 1)[0]
        # Binary sign quantization
        code = (feat > feat.median()).float()
    return code


img1 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0000.jpg"
img2 = "datasets/recognition_dataset/test/CASIA-Iris-Lamp/CASIA-Iris-Lamp_ts_0001.jpg"
img3 = "datasets/recognition_dataset/test/CASIA-Iris-Thousand/CASIA-Iris-Thousand_ts_0000.jpg"

b1 = get_binary_code(img1)
b2 = get_binary_code(img2)
b3 = get_binary_code(img3)

same_acc = (b1 == b2).float().mean().item()
diff_acc = (b1 == b3).float().mean().item()

print(f"Same Subject Binary Iris Code Accuracy     : {same_acc*100:.2f}%")
print(f"Different Subject Binary Iris Code Accuracy: {diff_acc*100:.2f}%")
