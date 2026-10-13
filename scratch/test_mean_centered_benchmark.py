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
m = IrisIdentityCNN(num_classes=2).to(device)
m.load_state_dict(torch.load(model_path, map_location=device))
m.eval()

tr = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_raw_vector(img_path):
    pil_img = Image.open(img_path).convert("RGB")
    t = tr(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = m.features(t)
        feat = m.avgpool(feat)
        feat = torch.flatten(feat, 1)[0]
    return feat


test_dir = os.path.abspath("datasets/recognition_dataset/test")
subjs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

all_feats = []
same_scores = []
diff_scores = []

# Collect sample features to compute dataset background mean vector
for subj in subjs[:5]:
    files = glob.glob(os.path.join(test_dir, subj, "*.jpg"))[:10]
    for f in files:
        all_feats.append(extract_raw_vector(f))

mean_vector = torch.stack(all_feats).mean(dim=0)


def get_centered_embedding(img_path):
    v = extract_raw_vector(img_path)
    centered = v - mean_vector
    return F.normalize(centered, p=2, dim=0)


# Evaluate Same-Subject Pairs
subj1_files = glob.glob(os.path.join(test_dir, subjs[0], "*.jpg"))
emb1_a = get_centered_embedding(subj1_files[0])
emb1_b = get_centered_embedding(subj1_files[1])
same_sim = torch.dot(emb1_a, emb1_b).item()

# Evaluate Different-Subject Pairs
subj2_files = glob.glob(os.path.join(test_dir, subjs[1], "*.jpg"))
emb2_a = get_centered_embedding(subj2_files[0])
diff_sim = torch.dot(emb1_a, emb2_a).item()

print(f"Same Subject Centered Similarity     : {same_sim:.4f} ({same_sim*100:.2f}%)")
print(f"Different Subject Centered Similarity: {diff_sim:.4f} ({diff_sim*100:.2f}%)")
