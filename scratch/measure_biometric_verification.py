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

sys.path.append(os.path.abspath("."))
from ai_models.iris_recognition.train_identity_metric_model_v2 import IrisIdentityMetricCNN_v2

class LocalCasiaDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.repo_root = os.path.abspath(".")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        subj_id, eye_side, orig_path, label_idx = self.data_list[idx]
        
        # Translate potential foreign path to local path
        if "datasets" in orig_path:
            rel_part = orig_path.split("datasets")[-1].lstrip("\\/")
            local_path = os.path.join(self.repo_root, "datasets", rel_part)
        else:
            local_path = orig_path

        img = Image.open(local_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label_idx, subj_id, eye_side, local_path

def evaluate_biometric_verification():
    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Biometric Verification on device: {device}")

    # Load Manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))

    # Strict subject-disjoint split: Unseen Test Subjects 900 to 999 (100 subjects)
    test_subjs = set(unique_subjs[900:])
    test_rows = [r for r in rows if r[0] in test_subjs]

    print(f"Total Unique Subjects in Dataset : {len(unique_subjs)}")
    print(f"Unseen Test Subjects (900-999)   : {len(test_subjs)} subjects ({len(test_rows)} images)")

    # Transform matching v2_embedding_pipeline
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = LocalCasiaDataset([(r[0], r[1], r[2], int(r[0])) for r in test_rows], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    embeddings = []
    labels = []
    eyes = []
    paths = []

    print("Extracting 512-D L2-normalized feature embeddings...")
    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in test_loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(subj_ids)
            eyes.extend(eye_sides)
            paths.extend(img_paths)

    embeddings = np.vstack(embeddings)
    total_imgs = len(embeddings)
    print(f"Extracted embeddings shape: {embeddings.shape}")

    # 1. Genuine Pairs: Same subject, same eye side, different images
    genuine_pairs = []
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
                sim = float(np.dot(embeddings[idx1], embeddings[idx2]))
                genuine_pairs.append(sim)

    # 2. Impostor Pairs: Different subjects, same eye side
    random.seed(42)
    indices_by_eye = {
        'L': [i for i, e in enumerate(eyes) if e == 'L'],
        'R': [i for i, e in enumerate(eyes) if e == 'R']
    }

    impostor_pairs = []
    target_impostors = 50000  # Large statistical sample of impostor comparisons
    while len(impostor_pairs) < target_impostors:
        eye_side = random.choice(['L', 'R'])
        pool = indices_by_eye[eye_side]
        i1, i2 = random.sample(pool, 2)
        if labels[i1] != labels[i2]:
            sim = float(np.dot(embeddings[i1], embeddings[i2]))
            impostor_pairs.append(sim)

    gen_sims = np.array(genuine_pairs)
    imp_sims = np.array(impostor_pairs)

    print(f"\nGenerated Pairs:")
    print(f"  Genuine Attempts  : {len(gen_sims):,}")
    print(f"  Impostor Attempts : {len(imp_sims):,}")

    # Calculate EER
    thresholds = np.linspace(-0.5, 1.0, 150001)
    fars = np.mean(imp_sims[:, None] >= thresholds[None, :], axis=0)
    frrs = np.mean(gen_sims[:, None] < thresholds[None, :], axis=0)
    eer_idx = np.argmin(np.abs(fars - frrs))
    eer_th = thresholds[eer_idx]
    eer_val = (fars[eer_idx] + frrs[eer_idx]) / 2.0 * 100.0

    print("\n==================================================")
    print(" BIOMETRIC VERIFICATION PERFORMANCE (EER OPERATING POINT)")
    print("==================================================")
    print(f"Equal Error Rate (EER) Threshold: {eer_th:.4f}")
    print(f"EER Value                       : {eer_val:.4f}%")
    print(f"FAR at EER                      : {fars[eer_idx]*100:.4f}%")
    print(f"FRR at EER                      : {frrs[eer_idx]*100:.4f}%")

    def report_operating_point(name, th):
        # Genuine attempts
        correct_accept = int(np.sum(gen_sims >= th))
        false_reject = int(np.sum(gen_sims < th))
        frr = (false_reject / len(gen_sims)) * 100.0

        # Impostor attempts
        correct_reject = int(np.sum(imp_sims < th))
        false_accept = int(np.sum(imp_sims >= th))
        far = (false_accept / len(imp_sims)) * 100.0

        total_attempts = len(gen_sims) + len(imp_sims)
        total_correct = correct_accept + correct_reject
        overall_accuracy = (total_correct / total_attempts) * 100.0

        print(f"\n--- OPERATING POINT: {name} (Threshold = {th:.4f}) ---")
        print(f"Genuine Evaluation:")
        print(f"  Total Genuine Attempts : {len(gen_sims):,}")
        print(f"  Correct Acceptances    : {correct_accept:,} (GAR = {100.0 - frr:.2f}%)")
        print(f"  False Rejections       : {false_reject:,} (FRR = {frr:.4f}%)")
        print(f"Impostor Evaluation:")
        print(f"  Total Impostor Attempts: {len(imp_sims):,}")
        print(f"  Correct Rejections     : {correct_reject:,} (GRR = {100.0 - far:.2f}%)")
        print(f"  False Acceptances      : {false_accept:,} (FAR = {far:.4f}%)")
        print(f"Overall Metrics:")
        print(f"  Total Test Attempts    : {total_attempts:,}")
        print(f"  Correct Decisions      : {total_correct:,}")
        print(f"  Overall Verification Accuracy : {overall_accuracy:.4f}%")
        return {
            "th": th,
            "gen_total": len(gen_sims),
            "correct_accept": correct_accept,
            "false_reject": false_reject,
            "frr": frr,
            "imp_total": len(imp_sims),
            "correct_reject": correct_reject,
            "false_accept": false_accept,
            "far": far,
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "accuracy": overall_accuracy
        }

    # Evaluate at EER threshold (0.3288) and at Default Application Threshold (0.80)
    report_operating_point("EER Optimal Operating Threshold", eer_th)
    report_operating_point("Production High-Security Threshold (0.8000)", 0.80)

if __name__ == "__main__":
    evaluate_biometric_verification()
