import os
import sys
import glob

sys.path.append(os.path.abspath("."))

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ai_models.iris_recognition.train_iris_recognition import IrisIdentityCNN

device = torch.device("cpu")
model_path = "ai_models/trained_models/iris_feature_model.pth"
model = IrisIdentityCNN(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

tr = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_feature_embedding(pil_img):
    tensor_img = tr(pil_img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.features(tensor_img)
        feat = model.avgpool(feat)
        feat = torch.flatten(feat, 1)
        normalized = F.normalize(feat, p=2, dim=1)[0]
    return normalized


test_dir = os.path.abspath("datasets/recognition_dataset/test")
subjs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

subj1_imgs = glob.glob(os.path.join(test_dir, subjs[0], "*.jpg"))
subj2_imgs = glob.glob(os.path.join(test_dir, subjs[1], "*.jpg"))

f1_a = extract_feature_embedding(Image.open(subj1_imgs[0]))
f1_b = extract_feature_embedding(Image.open(subj1_imgs[1]))
f2_a = extract_feature_embedding(Image.open(subj2_imgs[0]))

sim_same = F.cosine_similarity(f1_a.unsqueeze(0), f1_b.unsqueeze(0)).item()
sim_diff = F.cosine_similarity(f1_a.unsqueeze(0), f2_a.unsqueeze(0)).item()

print(f"Same Subject Feature Cosine Similarity     : {sim_same:.4f} ({sim_same*100:.2f}%)")
print(f"Different Subject Feature Cosine Similarity: {sim_diff:.4f} ({sim_diff*100:.2f}%)")
