import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from ai_models.iris_recognition.train_identity_metric_model import IrisIdentityMetricCNN, CasiaManifestDataset

torch.set_num_threads(8)

def evaluate_metric_model(num_pairs=5000):
    print("==================================================")
    print(" EVALUATING 512-D METRIC RECOGNITION MODEL (GENUINE vs IMPOSTOR)")
    print("==================================================")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model.pth")

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file '{model_path}' not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device : {device}")

    # Load model
    model = IrisIdentityMetricCNN(embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load manifest and extract test samples
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))
    subj_to_label = {s: i for i, s in enumerate(unique_subjs)}

    grouped = {}
    for r in rows:
        subj, eye, path = r[0], r[1], r[2]
        key = (subj, eye)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append((subj, eye, path, subj_to_label[subj]))

    test_samples = []
    for key, items in grouped.items():
        test_samples.extend(items[8:])  # Last 2 images of each (subject, eye)

    print(f"  Total Test Set Samples : {len(test_samples):,} images across 1,000 subjects")

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CasiaManifestDataset(test_samples, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    # Extract 512-D embeddings for all test samples with fast DataLoader batching
    embeddings = []
    labels = []
    eyes = []
    paths = []

    print("\nExtracting 512-D L2-Normalized Metric Embeddings in Batches...", flush=True)

    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in test_loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(lbls.numpy())
            eyes.extend(eye_sides)
            paths.extend(img_paths)

    embeddings = np.vstack(embeddings)

    # 1. SAME-EYE GENUINE PAIRS (Same subject, same eye side)
    genuine_sims_same_eye = []
    subj_eye_indices = {}
    for idx, (lbl, eye) in enumerate(zip(labels, eyes)):
        key = (lbl, eye)
        if key not in subj_eye_indices:
            subj_eye_indices[key] = []
        subj_eye_indices[key].append(idx)

    for key, indices in subj_eye_indices.items():
        if len(indices) >= 2:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    sim = np.dot(embeddings[indices[i]], embeddings[indices[j]])
                    genuine_sims_same_eye.append(sim)

    # 2. SAME-EYE IMPOSTOR PAIRS (Different subject, same eye side)
    impostor_sims_same_eye = []
    all_indices = list(range(len(labels)))

    random.seed(42)
    while len(impostor_sims_same_eye) < num_pairs:
        i, j = random.sample(all_indices, 2)
        if labels[i] != labels[j] and eyes[i] == eyes[j]:
            sim = np.dot(embeddings[i], embeddings[j])
            impostor_sims_same_eye.append(sim)

    # 3. CROSS-EYE PAIRS (Same subject, Left vs Right eye)
    cross_eye_sims = []
    subj_indices = {}
    for idx, lbl in enumerate(labels):
        if lbl not in subj_indices:
            subj_indices[lbl] = []
        subj_indices[lbl].append(idx)

    for lbl, indices in subj_indices.items():
        left_idx = [i for i in indices if eyes[i] == 'L']
        right_idx = [i for i in indices if eyes[i] == 'R']
        for li in left_idx:
            for ri in right_idx:
                sim = np.dot(embeddings[li], embeddings[ri])
                cross_eye_sims.append(sim)

    gen_same = np.array(genuine_sims_same_eye)
    imp_same = np.array(impostor_sims_same_eye)
    cross = np.array(cross_eye_sims)

    # Calculate ROC, FAR, FRR, EER and optimal threshold
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_eer = 1.0
    best_thresh = 0.50

    for th in thresholds:
        far = np.mean(imp_same >= th)
        frr = np.mean(gen_same < th)
        eer = (far + frr) / 2.0
        if abs(far - frr) < abs(best_eer - 0.0):
            best_eer = max(far, frr)
            best_thresh = th

    far_at_best = np.mean(imp_same >= best_thresh)
    frr_at_best = np.mean(gen_same < best_thresh)

    print(f"\n==================================================")
    print(f"      METRIC MODEL EVALUATION RESULTS (1,000 SUBJECTS)")
    print(f"==================================================")
    print(f"1. GENUINE PAIR SIMILARITIES (Same Subject, Same Eye):")
    print(f"   - Total Genuine Pairs Tested: {len(gen_same):,}")
    print(f"   - Minimum Genuine Similarity: {gen_same.min():.4f}")
    print(f"   - Maximum Genuine Similarity: {gen_same.max():.4f}")
    print(f"   - Mean Genuine Similarity   : {gen_same.mean():.4f} ± {gen_same.std():.4f}")

    print(f"\n2. IMPOSTOR PAIR SIMILARITIES (Different Subject, Same Eye):")
    print(f"   - Total Impostor Pairs Tested: {len(imp_same):,}")
    print(f"   - Minimum Impostor Similarity: {imp_same.min():.4f}")
    print(f"   - Maximum Impostor Similarity: {imp_same.max():.4f}")
    print(f"   - Mean Impostor Similarity  : {imp_same.mean():.4f} ± {imp_same.std():.4f}")

    print(f"\n3. CROSS-EYE SIMILARITIES (Same Subject, Left vs Right Eye):")
    print(f"   - Total Cross-Eye Pairs     : {len(cross):,}")
    print(f"   - Mean Cross-Eye Similarity : {cross.mean():.4f} ± {cross.std():.4f}")

    print(f"\n4. OPTIMAL VERIFICATION THRESHOLD & ERROR RATES:")
    print(f"   - Optimal Decision Threshold: Threshold = {best_thresh:.4f}")
    print(f"   - False Acceptance Rate (FAR): {far_at_best*100:.2f}%")
    print(f"   - False Rejection Rate  (FRR): {frr_at_best*100:.2f}%")
    print(f"   - Equal Error Rate      (EER): {best_eer*100:.2f}%")
    print(f"==================================================\n")


if __name__ == "__main__":
    evaluate_metric_model(num_pairs=5000)
