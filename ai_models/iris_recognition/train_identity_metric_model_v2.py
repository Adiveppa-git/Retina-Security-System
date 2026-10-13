import os
import sys
import math
import csv
import random
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.set_num_threads(8)


class ArcMarginProduct(nn.Module):
    """
    ArcFace Additive Angular Margin Loss Layer.
    s: norm scaling factor (30.0)
    m: additive angular margin (0.50 radians ~ 28.6 degrees)
    """
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


class IrisIdentityMetricCNN_v2(nn.Module):
    """
    512-Dimensional L2-Normalized Metric Feature Extractor v2.
    Backbone: MobileNetV3-Small pretrained on ImageNet.
    Output: 512-D L2-normalized feature embedding vector.
    """
    def __init__(self, embedding_dim=512):
        super(IrisIdentityMetricCNN_v2, self).__init__()
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


def get_iris_transforms():
    """
    Unified, reusable iris image preprocessing transforms.
    Resizes image to 64x256, converts to tensor, normalizes channel values.
    """
    train_transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def compute_set_verification_eer(model, loader, device):
    """
    Extracts embeddings for a dataset split and evaluates Genuine vs Impostor EER.
    """
    model.eval()
    embeddings = []
    labels = []
    eyes = []
    paths = []

    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(subj_ids)
            eyes.extend(eye_sides)
            paths.extend(img_paths)

    embeddings = np.vstack(embeddings)

    # Genuine pairs (same subject, same eye)
    genuine_sims = []
    subj_eye_map = {}
    for idx, (subj, eye) in enumerate(zip(labels, eyes)):
        key = (subj, eye)
        if key not in subj_eye_map:
            subj_eye_map[key] = []
        subj_eye_map[key].append(idx)

    for key, indices in subj_eye_map.items():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                sim = np.dot(embeddings[idx1], embeddings[idx2])
                genuine_sims.append(sim)

    # Impostor pairs (different subject, same eye)
    impostor_sims = []
    random.seed(42)
    indices_by_eye = {'L': [i for i, e in enumerate(eyes) if e == 'L'],
                      'R': [i for i, e in enumerate(eyes) if e == 'R']}

    target_count = min(5000, len(genuine_sims) * 3)
    while len(impostor_sims) < target_count:
        eye_side = random.choice(['L', 'R'])
        pool = indices_by_eye[eye_side]
        i1, i2 = random.sample(pool, 2)
        if labels[i1] != labels[i2]:
            sim = np.dot(embeddings[i1], embeddings[i2])
            impostor_sims.append(sim)

    gen_sims = np.array(genuine_sims)
    imp_sims = np.array(impostor_sims)

    fine_ths = np.linspace(0.0, 1.0, 10001)
    fars = np.mean(imp_sims[:, None] >= fine_ths[None, :], axis=0)
    frrs = np.mean(gen_sims[:, None] < fine_ths[None, :], axis=0)
    diffs = np.abs(fars - frrs)
    min_idx = np.argmin(diffs)

    eer_th = fine_ths[min_idx]
    eer_val = (fars[min_idx] + frrs[min_idx]) / 2.0
    return eer_val, eer_th, gen_sims, imp_sims


def train_v2_model(epochs=15, batch_size=256, lr=0.005):
    print("==================================================")
    print(" TRAINING SAFETY CONFIGURATION")
    print("==================================================")
    print("Dataset:")
    print("Train subjects = 800")
    print("Validation subjects = 100")
    print("Test subjects = 100\n")
    print("Images:")
    print("Train = 16,000")
    print("Validation = 2,000")
    print("Test = 2,000\n")
    print("Model:")
    print("Embedding = 512-D")
    print("ArcFace classes = 800\n")
    print("Training:")
    print(f"Epochs = {epochs}")
    print(f"Batch size = {batch_size}")
    print("Optimizer = AdamW")
    print(f"Learning rate = {lr}")
    print("Scheduler = CosineAnnealingLR")
    print("==================================================\n")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    v2_model_save_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2.pth")
    resume_checkpoint_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2_resume.pth")
    metrics_save_path = os.path.abspath("ai_models/trained_models/training_metrics_v2.json")

    if not os.path.exists(manifest_path):
        print(f"[ERROR] Manifest file '{manifest_path}' not found!")
        return

    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))

    # Subject-Disjoint Partitioning
    train_subjs = unique_subjs[:800]      # 800 Train Subjects (000 - 799)
    val_subjs   = unique_subjs[800:900]   # 100 Val Subjects   (800 - 899)
    test_subjs  = unique_subjs[900:]      # 100 Test Subjects  (900 - 999)

    train_subj_to_label = {s: i for i, s in enumerate(train_subjs)}

    train_samples = [(r[0], r[1], r[2], train_subj_to_label[r[0]]) for r in rows if r[0] in set(train_subjs)]
    val_samples   = [(r[0], r[1], r[2], int(r[0])) for r in rows if r[0] in set(val_subjs)]
    test_samples  = [(r[0], r[1], r[2], int(r[0])) for r in rows if r[0] in set(test_subjs)]

    print(f"  Total Subjects in Dataset : {len(unique_subjs)}")
    print(f"  Training Set              : {len(train_samples):,} images across {len(train_subjs)} subjects (Label range: 0 to {len(train_subjs)-1})")
    print(f"  Validation Set            : {len(val_samples):,} images across {len(val_subjs)} subjects [DISJOINT]")
    print(f"  Unseen Test Set           : {len(test_samples):,} images across {len(test_subjs)} subjects [DISJOINT]")

    # Verify zero data leakage
    train_paths = set([s[2] for s in train_samples])
    val_paths = set([s[2] for s in val_samples])
    test_paths = set([s[2] for s in test_samples])

    assert len(train_paths.intersection(val_paths)) == 0, "LEAKAGE: Train & Val overlap!"
    assert len(train_paths.intersection(test_paths)) == 0, "LEAKAGE: Train & Test overlap!"
    assert len(val_paths.intersection(test_paths)) == 0, "LEAKAGE: Val & Test overlap!"
    print("  Data Leakage Audit        : 100% VERIFIED ZERO DATA LEAKAGE!\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device          : {device}")

    train_transform, val_test_transform = get_iris_transforms()

    train_dataset = CasiaManifestDataset(train_samples, transform=train_transform)
    val_dataset   = CasiaManifestDataset(val_samples, transform=val_test_transform)
    test_dataset  = CasiaManifestDataset(test_samples, transform=val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Instantiate Model & ArcFace Metric Layer (800 target classes)
    embedding_model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
    arc_metric = ArcMarginProduct(in_features=512, out_features=len(train_subjs), s=30.0, m=0.50).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(list(embedding_model.parameters()) + list(arc_metric.parameters()), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    start_epoch = 1
    best_val_eer = 1.0
    best_epoch = 0
    training_history = []

    if os.path.exists(resume_checkpoint_path):
        print(f"\n[RESUME CHECKPOINT DETECTED] Found '{resume_checkpoint_path}'!")
        try:
            ckpt = torch.load(resume_checkpoint_path, map_location=device)
            embedding_model.load_state_dict(ckpt["model_state_dict"])
            arc_metric.load_state_dict(ckpt["arc_metric_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_eer = ckpt.get("best_val_eer", 1.0)
            best_epoch = ckpt.get("best_epoch", 0)
            training_history = ckpt.get("training_history", [])
            print(f"[RESUME SUCCESS] Safe state restored! Resuming from Epoch {start_epoch} of {epochs}.\n", flush=True)
        except Exception as e:
            print(f"[RESUME ERROR] Could not parse resume checkpoint ({e}). Starting clean from Epoch 1.\n", flush=True)
            start_epoch = 1
    else:
        print("\nStarting Clean Training Run from Epoch 1...\n", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        start_t = time.time()
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
                print(f"   Epoch [{epoch:02d}/{epochs:02d}] | Batch [{i+1:03d}/{len(train_loader):03d}] | Loss: {(running_loss/total):.4f} | Acc: {(correct/total)*100.0:.2f}%", flush=True)

        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluate Validation Set EER after each epoch
        val_eer, val_eer_th, val_gen, val_imp = compute_set_verification_eer(embedding_model, val_loader, device)

        epoch_time = time.time() - start_t
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.1f}s) | LR: {curr_lr:.6f} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: N/A (Disjoint Val Set) | Val EER: {val_eer*100:.2f}% (th={val_eer_th:.4f}) | Best Val EER: {best_val_eer*100:.2f}%", flush=True)

        epoch_info = {
            "epoch": epoch,
            "lr": curr_lr,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_eer": float(val_eer * 100.0),
            "val_eer_th": float(val_eer_th)
        }
        training_history.append(epoch_info)

        # Checkpoint Best Validation Model
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_epoch = epoch
            os.makedirs(os.path.dirname(v2_model_save_path), exist_ok=True)
            torch.save(embedding_model.state_dict(), v2_model_save_path)
            print(f"   ---> [BEST CHECKPOINT SAVED] Val EER improved to {val_eer*100:.2f}% at epoch {epoch}!", flush=True)

        # Save Safe Epoch Resume Checkpoint
        os.makedirs(os.path.dirname(resume_checkpoint_path), exist_ok=True)
        resume_state = {
            "epoch": epoch,
            "model_state_dict": embedding_model.state_dict(),
            "arc_metric_state_dict": arc_metric.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_eer": float(best_val_eer),
            "best_epoch": int(best_epoch),
            "training_history": training_history,
            "seed": SEED,
            "config": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "embedding_dim": 512,
                "arc_s": 30.0,
                "arc_m": 0.50
            }
        }
        torch.save(resume_state, resume_checkpoint_path)
        print(f"   ---> [RESUME CHECKPOINT SAVED] Safe resume state saved for epoch {epoch} to '{resume_checkpoint_path}'.", flush=True)

    print("\n==================================================")
    print(f" TRAINING COMPLETE! Best Checkpoint Saved at Epoch {best_epoch} (Val EER: {best_val_eer*100:.2f}%)")
    print("==================================================\n")

    # Load Best Model for Final Unseen Test Evaluation
    print("Loading Best Checkpoint for Final Unseen Test Evaluation (Subjects 900-999)...")
    best_model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
    best_model.load_state_dict(torch.load(v2_model_save_path, map_location=device))

    test_eer, test_eer_th, test_gen, test_imp = compute_set_verification_eer(best_model, test_loader, device)

    # Diagnostic Outliers Extraction
    best_model.eval()
    test_embs = []
    test_lbls = []
    test_eyess = []
    test_ptss = []

    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in test_loader:
            imgs = imgs.to(device)
            embs = best_model(imgs).cpu().numpy()
            test_embs.append(embs)
            test_lbls.extend(subj_ids)
            test_eyess.extend(eye_sides)
            test_ptss.extend(img_paths)

    test_embs = np.vstack(test_embs)

    gen_pairs_detailed = []
    subj_eye_map = {}
    for idx, (subj, eye) in enumerate(zip(test_lbls, test_eyess)):
        key = (subj, eye)
        if key not in subj_eye_map:
            subj_eye_map[key] = []
        subj_eye_map[key].append(idx)

    for key, indices in subj_eye_map.items():
        subj, eye = key
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                i1, i2 = indices[i], indices[j]
                sim = np.dot(test_embs[i1], test_embs[i2])
                gen_pairs_detailed.append({
                    "subj": subj,
                    "eye": eye,
                    "p1": test_ptss[i1],
                    "p2": test_ptss[i2],
                    "sim": float(sim)
                })

    imp_pairs_detailed = []
    indices_by_eye = {'L': [i for i, e in enumerate(test_eyess) if e == 'L'],
                      'R': [i for i, e in enumerate(test_eyess) if e == 'R']}

    random.seed(42)
    while len(imp_pairs_detailed) < 10000:
        eye_side = random.choice(['L', 'R'])
        pool = indices_by_eye[eye_side]
        i1, i2 = random.sample(pool, 2)
        if test_lbls[i1] != test_lbls[i2]:
            sim = np.dot(test_embs[i1], test_embs[i2])
            imp_pairs_detailed.append({
                "subj1": test_lbls[i1],
                "subj2": test_lbls[i2],
                "eye": eye_side,
                "p1": test_ptss[i1],
                "p2": test_ptss[i2],
                "sim": float(sim)
            })

    lowest_20_gen = sorted(gen_pairs_detailed, key=lambda x: x['sim'])[:20]
    highest_20_imp = sorted(imp_pairs_detailed, key=lambda x: x['sim'], reverse=True)[:20]

    # Save diagnostic outliers to directory
    diag_dir = os.path.abspath("ai_models/iris_recognition/diagnostics_v2")
    os.makedirs(diag_dir, exist_ok=True)

    with open(os.path.join(diag_dir, "lowest_20_genuine.json"), "w") as f:
        json.dump(lowest_20_gen, f, indent=2)

    with open(os.path.join(diag_dir, "highest_20_impostor.json"), "w") as f:
        json.dump(highest_20_imp, f, indent=2)

    # Prepare complete final JSON result dump
    results = {
        "best_epoch": best_epoch,
        "best_val_eer": float(best_val_eer * 100.0),
        "test_eer": float(test_eer * 100.0),
        "test_eer_th": float(test_eer_th),
        "training_history": training_history,
        "gen_stats": {
            "mean": float(test_gen.mean()),
            "std": float(test_gen.std()),
            "median": float(np.median(test_gen)),
            "min": float(test_gen.min()),
            "max": float(test_gen.max()),
            "percentiles": [float(x) for x in np.percentile(test_gen, [1, 5, 25, 50, 75, 95, 99])]
        },
        "imp_stats": {
            "mean": float(test_imp.mean()),
            "std": float(test_imp.std()),
            "median": float(np.median(test_imp)),
            "min": float(test_imp.min()),
            "max": float(test_imp.max()),
            "percentiles": [float(x) for x in np.percentile(test_imp, [1, 5, 25, 50, 75, 95, 99])]
        },
        "lowest_20_gen": lowest_20_gen,
        "highest_20_imp": highest_20_imp
    }

    with open(metrics_save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n==================================================")
    print(f" UNSEEN TEST EVALUATION RESULTS (100 UNSEEN SUBJECTS)")
    print(f"==================================================")
    print(f"  - Best Checkpoint Epoch : {best_epoch}")
    print(f"  - Validation EER        : {best_val_eer*100:.2f}%")
    print(f"  - Unseen Test EER       : {test_eer*100:.2f}% (at threshold {test_eer_th:.4f})")
    print(f"  - Genuine Sim (Mean±Std): {test_gen.mean():.4f} ± {test_gen.std():.4f} | Median: {np.median(test_gen):.4f}")
    print(f"  - Impostor Sim(Mean±Std): {test_imp.mean():.4f} ± {test_imp.std():.4f} | Median: {np.median(test_imp):.4f}")
    print(f"==================================================\n")

if __name__ == "__main__":
    train_v2_model(epochs=15, batch_size=256, lr=0.005)
