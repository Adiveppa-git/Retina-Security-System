import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_feat, label):
        cosine = F.linear(F.normalize(input_feat, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input_feat.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

class IrisIdentityMetricCNN_v3(nn.Module):
    def __init__(self, embedding_dim=512):
        super(IrisIdentityMetricCNN_v3, self).__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(576, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def extract_embedding(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x):
        return self.extract_embedding(x)

class RGBEyeDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.transform = transform
        self.samples = []
        files = os.listdir(folder_path)
        for fname in files:
            if not fname.endswith(".png") and not fname.endswith(".jpg"):
                continue
            path = os.path.join(folder_path, fname)
            # Example filename: u0_Renkigouda_L_sess1_0.png
            parts = fname.split("_")
            user_id = int(parts[0].replace("u", ""))
            self.samples.append((path, user_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path

def train_rgb_v3_model():
    print("==================================================")
    print(" EXECUTING RGB-DOMAIN FINE-TUNING EXPERIMENT (V3)")
    print("==================================================")

    train_dir = os.path.abspath("dataset/rgb_webcam_dataset/train")
    val_dir = os.path.abspath("dataset/rgb_webcam_dataset/val")
    save_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_rgb_v3.pth")

    transform_train = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = RGBEyeDataset(train_dir, transform=transform_train)
    val_ds = RGBEyeDataset(val_dir, transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = IrisIdentityMetricCNN_v3(embedding_dim=512).to(device)

    # Load base V2 checkpoint weights
    v2_ckpt = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2.pth")
    if os.path.exists(v2_ckpt):
        model.load_state_dict(torch.load(v2_ckpt, map_location=device))
        print(f"Loaded base V2 model weights from: {v2_ckpt}")

    arc_head = ArcMarginProduct(in_features=512, out_features=3, s=30.0, m=0.50).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(arc_head.parameters()), lr=1e-4, weight_decay=1e-4)

    # Fine-tuning loop (15 epochs)
    model.train()
    arc_head.train()
    epochs = 15

    print(f"\nFine-tuning model on {len(train_ds)} train samples across {epochs} epochs...")
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embs = model(imgs)
            outputs = arc_head(embs, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_ds)
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f" Epoch [{epoch+1:2d}/{epochs}] Loss = {epoch_loss:.4f}")

    # Save RGB V3 model checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n[SUCCESS] RGB Fine-Tuned Model V3 saved to: {save_path}")

    # INDEPENDENT EVALUATION & DISTRIBUTIONS
    model.eval()
    val_embs = []
    val_labels = []
    val_paths = []

    with torch.no_grad():
        for img, label, path in val_loader:
            img = img.to(device)
            emb = model(img)
            val_embs.append(emb.cpu().squeeze(0))
            val_labels.append(label.item())
            val_paths.append(path[0])

    genuine_sims = []
    impostor_sims = []

    for i in range(len(val_embs)):
        for j in range(len(val_embs)):
            if i < j:
                sim = float(F.cosine_similarity(val_embs[i].unsqueeze(0), val_embs[j].unsqueeze(0)).item())
                if val_labels[i] == val_labels[j]:
                    genuine_sims.append(sim)
                else:
                    impostor_sims.append(sim)

    gen_mean = float(np.mean(genuine_sims)) if genuine_sims else 0.0
    gen_std = float(np.std(genuine_sims)) if genuine_sims else 0.0
    gen_min = float(np.min(genuine_sims)) if genuine_sims else 0.0
    gen_max = float(np.max(genuine_sims)) if genuine_sims else 0.0

    imp_mean = float(np.mean(impostor_sims)) if impostor_sims else 0.0
    imp_std = float(np.std(impostor_sims)) if impostor_sims else 0.0
    imp_min = float(np.min(impostor_sims)) if impostor_sims else 0.0
    imp_max = float(np.max(impostor_sims)) if impostor_sims else 0.0

    # Calculate FAR, FRR, EER at optimal threshold
    thresholds = np.linspace(-1.0, 1.0, 201)
    eer = 1.0
    opt_thresh = 0.3288
    min_diff = 1.0

    for th in thresholds:
        frr = np.mean([1.0 if s < th else 0.0 for s in genuine_sims]) if genuine_sims else 0.0
        far = np.mean([1.0 if s >= th else 0.0 for s in impostor_sims]) if impostor_sims else 0.0
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            eer = (far + frr) / 2.0
            opt_thresh = float(th)

    frr_opt = float(np.mean([1.0 if s < opt_thresh else 0.0 for s in genuine_sims]))
    far_opt = float(np.mean([1.0 if s >= opt_thresh else 0.0 for s in impostor_sims]))

    print("\n--------------------------------------------------")
    print(" RGB V3 MODEL EVALUATION REPORT & METRICS")
    print("--------------------------------------------------")
    print(f" Genuine Pair Cosine Similarities ({len(genuine_sims)} pairs):")
    print(f"   Mean = {gen_mean:.4f} | Std = {gen_std:.4f} | Min = {gen_min:.4f} | Max = {gen_max:.4f}")
    print(f" Impostor Pair Cosine Similarities ({len(impostor_sims)} pairs):")
    print(f"   Mean = {imp_mean:.4f} | Std = {imp_std:.4f} | Min = {imp_min:.4f} | Max = {imp_max:.4f}")
    print(f" Optimal RGB Threshold             : {opt_thresh:.4f}")
    print(f" False Accept Rate (FAR)          : {far_opt*100.2:.2f}% ({far_opt:.4f})")
    print(f" False Reject Rate (FRR)          : {frr_opt*100.0:.2f}% ({frr_opt:.4f})")
    print(f" Equal Error Rate (EER)           : {eer*100.0:.2f}% ({eer:.4f})")
    print("--------------------------------------------------\n")

if __name__ == "__main__":
    train_rgb_v3_model()
