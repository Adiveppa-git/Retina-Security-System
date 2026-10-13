import os
import sys
import math
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


class ArcMarginProduct(nn.Module):
    """ArcFace Additive Angular Margin Loss Layer for Subject Metric Learning."""
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


class IrisIdentityMetricCNN(nn.Module):
    """
    512-Dimensional L2-Normalized Metric Feature Extractor.
    Backbone: MobileNetV3-Small
    """
    def __init__(self, embedding_dim=512):
        super(IrisIdentityMetricCNN, self).__init__()
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


class CasiaManifestDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subj_id, eye_side, img_path, label_idx = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label_idx, subj_id, eye_side, img_path


def train_metric_model(num_subjects=200, max_epochs=2, batch_size=128, lr=0.003):
    print("==================================================")
    print(" PHASE 2: TRAINING 512-D ARCFACE METRIC IDENTITY MODEL")
    print("==================================================")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_save_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model.pth")

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest file '{manifest_path}' not found!")
        return

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))[:num_subjects]
    subj_to_label = {s: i for i, s in enumerate(unique_subjs)}

    grouped = {}
    for r in rows:
        subj, eye, path = r[0], r[1], r[2]
        if subj in subj_to_label:
            key = (subj, eye)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append((subj, eye, path, subj_to_label[subj]))

    train_samples = []
    test_samples = []

    for key, items in grouped.items():
        train_samples.extend(items[:8])
        test_samples.extend(items[8:])

    print(f"  Total Subject Identities : {len(unique_subjs):,} classes ({unique_subjs[0]} to {unique_subjs[-1]})")
    print(f"  Training Split           : {len(train_samples):,} images (16 per subject)")
    print(f"  Testing Split            : {len(test_samples):,} images (4 per subject)")

    train_paths = set([s[2] for s in train_samples])
    test_paths = set([s[2] for s in test_samples])
    overlap = train_paths.intersection(test_paths)
    print(f"  Train/Test Image Overlap Check: {len(overlap)} duplicate images (0 LEAKAGE VERIFIED!)")
    assert len(overlap) == 0, "DATA LEAKAGE DETECTED!"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device         : {device}")

    transform_train = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CasiaManifestDataset(train_samples, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    embedding_model = IrisIdentityMetricCNN(embedding_dim=512).to(device)
    arc_metric = ArcMarginProduct(in_features=512, out_features=len(unique_subjs), s=30.0, m=0.50).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(embedding_model.parameters()) + list(arc_metric.parameters()), lr=lr, weight_decay=1e-4)

    print("\nStarting Metric Learning Training Epochs...", flush=True)

    best_loss = float("inf")

    for epoch in range(1, max_epochs + 1):
        embedding_model.train()
        arc_metric.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels, _, _, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = embedding_model(images)
            outputs = arc_metric(embeddings, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{max_epochs}] | Batch [{i+1}/{len(train_loader)}] | ArcFace Loss: {(running_loss/total):.4f}", flush=True)

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0

        print(f"  EPOCH {epoch} COMPLETED | Loss: {epoch_loss:.4f} | Margin Acc: {epoch_acc:.2f}%\n", flush=True)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(embedding_model.state_dict(), model_save_path)
            print(f"  ---> [Checkpoint Saved] Saved Metric Model to {model_save_path}\n", flush=True)

    print("==================================================")
    print(" METRIC MODEL TRAINING COMPLETE SUCCESS!")
    print("==================================================\n")


if __name__ == "__main__":
    train_metric_model(num_subjects=200, max_epochs=2, batch_size=128, lr=0.003)
