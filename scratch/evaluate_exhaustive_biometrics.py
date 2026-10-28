import os
import sys
import csv
import json
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
        if "datasets" in orig_path:
            rel_part = orig_path.split("datasets")[-1].lstrip("\\/")
            local_path = os.path.join(self.repo_root, "datasets", rel_part)
        else:
            local_path = orig_path

        img = Image.open(local_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label_idx, subj_id, eye_side, local_path

def run_exhaustive_verification_audit():
    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))
    test_subjs = set(unique_subjs[900:]) # 900 to 999 (100 unseen subjects)
    test_rows = [r for r in rows if r[0] in test_subjs]

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = LocalCasiaDataset([(r[0], r[1], r[2], int(r[0])) for r in test_rows], transform=transform)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    embeddings = []
    labels = []
    eyes = []

    print("Extracting embeddings for 2,000 test images...")
    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, _ in test_loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(subj_ids)
            eyes.extend(eye_sides)

    embeddings = np.vstack(embeddings)
    labels = np.array(labels)
    eyes = np.array(eyes)

    # Separate Left and Right Eye subsets
    left_mask = (eyes == "L")
    right_mask = (eyes == "R")

    left_embs = embeddings[left_mask]
    left_labels = labels[left_mask]

    right_embs = embeddings[right_mask]
    right_labels = labels[right_mask]

    print(f"Left images: {len(left_embs)}, Right images: {len(right_embs)}")

    def compute_all_pairs(embs, lbls):
        sim_matrix = np.dot(embs, embs.T)
        N = len(embs)
        triu_indices = np.triu_indices(N, k=1)
        sims = sim_matrix[triu_indices]
        
        lbl_i = lbls[triu_indices[0]]
        lbl_j = lbls[triu_indices[1]]
        
        is_genuine = (lbl_i == lbl_j)
        gen = sims[is_genuine]
        imp = sims[~is_genuine]
        return gen, imp

    left_gen, left_imp = compute_all_pairs(left_embs, left_labels)
    right_gen, right_imp = compute_all_pairs(right_embs, right_labels)

    all_gen = np.sort(np.concatenate([left_gen, right_gen]))
    all_imp = np.sort(np.concatenate([left_imp, right_imp]))

    total_gen = len(all_gen)
    total_imp = len(all_imp)

    print(f"\n==================================================")
    print(f" EXHAUSTIVE PAIRWISE EVALUATION (NO SUBSAMPLING)")
    print(f"==================================================")
    print(f"Total Exhaustive Genuine Comparisons: {total_gen:,}")
    print(f"Total Exhaustive Impostor Comparisons: {total_imp:,}")
    print(f"Total Biometric Comparisons         : {total_gen + total_imp:,}")

    # Use np.searchsorted on sorted arrays for memory-free exact EER
    thresholds = np.linspace(-0.5, 1.0, 150001)
    
    # imp >= th is total_imp - searchsorted(all_imp, th, side='left')
    imp_ge_th = total_imp - np.searchsorted(all_imp, thresholds, side='left')
    fars = imp_ge_th / total_imp

    # gen < th is searchsorted(all_gen, th, side='left')
    gen_lt_th = np.searchsorted(all_gen, thresholds, side='left')
    frrs = gen_lt_th / total_gen

    eer_idx = np.argmin(np.abs(fars - frrs))
    eer_th = float(thresholds[eer_idx])
    eer_val = float((fars[eer_idx] + frrs[eer_idx]) / 2.0 * 100.0)

    print(f"\nExact Equal Error Rate (EER): {eer_val:.4f}% at threshold tau = {eer_th:.4f}")
    print(f"  FAR at EER: {fars[eer_idx]*100:.4f}% ({imp_ge_th[eer_idx]:,}/{total_imp:,})")
    print(f"  FRR at EER: {frrs[eer_idx]*100:.4f}% ({gen_lt_th[eer_idx]:,}/{total_gen:,})")

    # Detailed Threshold Table
    threshold_evals = {}
    for th in [eer_th, 0.3288, 0.40, 0.50, 0.60, 0.70, 0.80, 0.82]:
        false_accept = int(total_imp - np.searchsorted(all_imp, th, side='left'))
        corr_reject = int(total_imp - false_accept)
        false_reject = int(np.searchsorted(all_gen, th, side='left'))
        corr_accept = int(total_gen - false_reject)

        frr = (false_reject / total_gen) * 100.0
        far = (false_accept / total_imp) * 100.0
        tot_attempts = total_gen + total_imp
        tot_correct = corr_accept + corr_reject
        acc = (tot_correct / tot_attempts) * 100.0

        threshold_evals[f"{th:.4f}"] = {
            "threshold": th,
            "genuine_attempts": total_gen,
            "correct_acceptances": corr_accept,
            "false_rejections": false_reject,
            "frr_pct": round(frr, 4),
            "impostor_attempts": total_imp,
            "correct_rejections": corr_reject,
            "false_acceptances": false_accept,
            "far_pct": round(far, 4),
            "total_attempts": tot_attempts,
            "total_correct": tot_correct,
            "accuracy_pct": round(acc, 4)
        }
        print(f"\nThreshold tau = {th:.4f}:")
        print(f"  Accuracy: {acc:.4f}% ({tot_correct:,} / {tot_attempts:,})")
        print(f"  FAR: {far:.4f}% ({false_accept:,} / {total_imp:,})")
        print(f"  FRR: {frr:.4f}% ({false_reject:,} / {total_gen:,})")
        print(f"  GAR: {100.0 - frr:.4f}% ({corr_accept:,} / {total_gen:,})")

    results_payload = {
        "dataset": "CASIA-Iris-Thousand (Subjects 900-999)",
        "unseen_subjects_count": 100,
        "images_count": len(embeddings),
        "total_genuine_pairs": total_gen,
        "total_impostor_pairs": total_imp,
        "total_pairwise_comparisons": total_gen + total_imp,
        "eer_threshold": eer_th,
        "eer_value_pct": eer_val,
        "threshold_evaluations": threshold_evals
    }

    out_file = os.path.abspath("scratch/biometric_exhaustive_verified_metrics.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nExhaustive verification results saved to: {out_file}")

if __name__ == "__main__":
    run_exhaustive_verification_audit()
