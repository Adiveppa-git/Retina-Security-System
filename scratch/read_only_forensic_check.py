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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ai_models.iris_recognition.train_identity_metric_model import IrisIdentityMetricCNN, CasiaManifestDataset

torch.set_num_threads(8)

def run_forensic_check():
    print("==================================================")
    print(" FORENSIC DIAGNOSTIC AUDIT & VERIFICATION SCRIPT")
    print("==================================================")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model.pth")

    # 1. CHECKPOINT TENSOR INSPECTION
    print("\n--- 1. CHECKPOINT TENSOR INSPECTION ---")
    state_dict = torch.load(model_path, map_location="cpu")
    print("Keys in state_dict:")
    for k, v in state_dict.items():
        print(f"  {k}: shape {v.shape}")

    mtime = os.path.getmtime(model_path)
    import datetime
    print(f"Model Checkpoint Modification Time: {datetime.datetime.fromtimestamp(mtime)}")

    # 2. RUN EMBEDDING EXTRACTION ON UNSEEN TEST SUBJECTS (900 - 999)
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))
    test_subjs = unique_subjs[900:] # 100 unseen test subjects

    test_samples = [r for r in rows if r[0] in set(test_subjs)]

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CasiaManifestDataset([(r[0], r[1], r[2], int(r[0])) for r in test_samples], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IrisIdentityMetricCNN(embedding_dim=512).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    embeddings = []
    labels = []
    eyes = []
    paths = []

    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in test_loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(subj_ids)
            eyes.extend(eye_sides)
            paths.extend(img_paths)

    embeddings = np.vstack(embeddings)

    # Pair generation
    genuine_pairs = []
    subj_eye_map = {}
    for idx, (subj, eye) in enumerate(zip(labels, eyes)):
        key = (subj, eye)
        if key not in subj_eye_map:
            subj_eye_map[key] = []
        subj_eye_map[key].append(idx)

    for key, indices in subj_eye_map.items():
        subj, eye = key
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                sim = np.dot(embeddings[idx1], embeddings[idx2])
                genuine_pairs.append({
                    "subj": subj,
                    "eye": eye,
                    "idx1": idx1,
                    "idx2": idx2,
                    "p1": paths[idx1],
                    "p2": paths[idx2],
                    "sim": float(sim)
                })

    impostor_pairs = []
    random.seed(42)
    indices_by_eye = {'L': [i for i, e in enumerate(eyes) if e == 'L'],
                      'R': [i for i, e in enumerate(eyes) if e == 'R']}

    target_impostor_count = 10000
    while len(impostor_pairs) < target_impostor_count:
        eye_side = random.choice(['L', 'R'])
        pool = indices_by_eye[eye_side]
        i1, i2 = random.sample(pool, 2)
        if labels[i1] != labels[i2]:
            sim = np.dot(embeddings[i1], embeddings[i2])
            impostor_pairs.append({
                "subj1": labels[i1],
                "subj2": labels[i2],
                "idx1": i1,
                "idx2": i2,
                "eye": eye_side,
                "p1": paths[i1],
                "p2": paths[i2],
                "sim": float(sim)
            })

    gen_sims = np.array([p['sim'] for p in genuine_pairs])
    imp_sims = np.array([p['sim'] for p in impostor_pairs])

    # 2. EXACT MATHEMATICAL EER RECALCULATION
    print("\n--- 2. EER RECALCULATION ---")
    fine_ths = np.linspace(0.0, 1.0, 100001)
    fars = np.mean(imp_sims[:, None] >= fine_ths[None, :], axis=0)
    frrs = np.mean(gen_sims[:, None] < fine_ths[None, :], axis=0)

    diffs = np.abs(fars - frrs)
    min_idx = np.argmin(diffs)

    eer_th = fine_ths[min_idx]
    far_at_eer = fars[min_idx]
    frr_at_eer = frrs[min_idx]
    eer_val = (far_at_eer + frr_at_eer) / 2.0

    print(f"Exact EER Threshold : {eer_th:.4f}")
    print(f"FAR at EER          : {far_at_eer*100:.4f}%")
    print(f"FRR at EER          : {frr_at_eer*100:.4f}%")
    print(f"EER Value           : {eer_val*100:.4f}%")

    # 3. STATISTICAL PERCENTILES & COLLAPSE ANALYSIS
    print("\n--- 3. DETAILED DISTRIBUTION PERCENTILES ---")
    print(f"Genuine  - Mean: {gen_sims.mean():.4f}, Std: {gen_sims.std():.4f}, Median: {np.median(gen_sims):.4f}")
    print(f"Impostor - Mean: {imp_sims.mean():.4f}, Std: {imp_sims.std():.4f}, Median: {np.median(imp_sims):.4f}")

    gen_p = np.percentile(gen_sims, [1, 5, 25, 50, 75, 95, 99])
    imp_p = np.percentile(imp_sims, [1, 5, 25, 50, 75, 95, 99])

    print(f"Genuine  Percentiles [P1, P5, P25, P50, P75, P95, P99]: {np.round(gen_p, 4)}")
    print(f"Impostor Percentiles [P1, P5, P25, P50, P75, P95, P99]: {np.round(imp_p, 4)}")

    norms = np.linalg.norm(embeddings, axis=1)
    var_per_dim = np.var(embeddings, axis=0)
    print(f"Mean Vector Norm: {norms.mean():.6f} (Std: {norms.std():.6f})")
    print(f"Per-dimension Variance - Mean: {var_per_dim.mean():.6f}, Min: {var_per_dim.min():.6f}, Max: {var_per_dim.max():.6f}")

    # 4. PAIR GENERATION INTEGRITY CHECK
    print("\n--- 4. PAIR GENERATION INTEGRITY CHECK ---")
    same_img_gen = sum(1 for p in genuine_pairs if p['idx1'] == p['idx2'])
    same_subj_imp = sum(1 for p in impostor_pairs if p['subj1'] == p['subj2'])
    eye_mismatch_gen = sum(1 for p in genuine_pairs if eyes[p['idx1']] != eyes[p['idx2']])
    eye_mismatch_imp = sum(1 for p in impostor_pairs if eyes[p['idx1']] != eyes[p['idx2']])

    print(f"Genuine pairs using same image: {same_img_gen}")
    print(f"Impostor pairs using same subject: {same_subj_imp}")
    print(f"Genuine eye side mismatch: {eye_mismatch_gen}")
    print(f"Impostor eye side mismatch: {eye_mismatch_imp}")

    # 5. FRESH THRESHOLD TABLE
    print("\n--- 5. FRESH THRESHOLD TABLE ---")
    print(f"{'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'GAR (%)':<10} | {'IRR (%)':<10}")
    print("-" * 60)
    table_ths = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96]
    for th in table_ths:
        far = np.mean(imp_sims >= th) * 100.0
        frr = np.mean(gen_sims < th) * 100.0
        gar = 100.0 - frr
        irr = 100.0 - far
        print(f"{th:<10.2f} | {far:<10.2f} | {frr:<10.2f} | {gar:<10.2f} | {irr:<10.2f}")

    # FAR <= 1.0% and FAR <= 0.1%
    idx_far_1 = np.where(fars <= 0.01)[0][0]
    idx_far_01 = np.where(fars <= 0.001)[0][0]
    print(f"\nThreshold at FAR <= 1.0%: {fine_ths[idx_far_1]:.4f} | FAR: {fars[idx_far_1]*100:.2f}% | FRR: {frrs[idx_far_1]*100:.2f}%")
    print(f"Threshold at FAR <= 0.1%: {fine_ths[idx_far_01]:.4f} | FAR: {fars[idx_far_01]*100:.2f}% | FRR: {frrs[idx_far_01]*100:.2f}%")

if __name__ == "__main__":
    run_forensic_check()
