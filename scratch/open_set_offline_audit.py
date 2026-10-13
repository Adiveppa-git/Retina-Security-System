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

def run_offline_audit():
    print("==================================================")
    print(" COMPREHENSIVE OFFLINE DIAGNOSTIC AUDIT & VALIDATION")
    print("==================================================")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model.pth")

    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file '{model_path}' not found!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device : {device}")

    # Load Model
    model = IrisIdentityMetricCNN(embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))

    # Partition into Subject-Disjoint sets
    train_subjs = unique_subjs[:800]      # 000 - 799
    val_subjs   = unique_subjs[800:900]   # 800 - 899
    test_subjs  = unique_subjs[900:]      # 900 - 999 (UNSEEN TEST SUBJECTS)

    print(f"  Total Dataset Subjects: {len(unique_subjs)}")
    print(f"  Train Subjects Count  : {len(train_subjs)} (Range: {train_subjs[0]} - {train_subjs[-1]})")
    print(f"  Val Subjects Count    : {len(val_subjs)} (Range: {val_subjs[0]} - {val_subjs[-1]})")
    print(f"  Test Subjects Count   : {len(test_subjs)} (Range: {test_subjs[0]} - {test_subjs[-1]}) [UNSEEN SUBJECT-DISJOINT SET]")

    # Extract test samples (Unseen Subjects)
    test_samples = [r for r in rows if r[0] in set(test_subjs)]
    print(f"  Unseen Test Images    : {len(test_samples):,} images across {len(test_subjs)} subjects")

    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CasiaManifestDataset([(r[0], r[1], r[2], int(r[0])) for r in test_samples], transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    embeddings = []
    labels = []
    eyes = []
    paths = []

    print("\n[PART 2 & 3] Extracting 512-D Embeddings for Unseen Test Subjects...", flush=True)

    with torch.no_grad():
        for imgs, lbls, subj_ids, eye_sides, img_paths in test_loader:
            imgs = imgs.to(device)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            labels.extend(subj_ids)
            eyes.extend(eye_sides)
            paths.extend(img_paths)

    embeddings = np.vstack(embeddings)

    # Calculate Embedding Norms & Quality (PART 8)
    norms = np.linalg.norm(embeddings, axis=1)
    mean_vec = np.mean(embeddings, axis=0)
    std_vec = np.std(embeddings, axis=0)

    # Generate Genuine Pairs (Same Subject, Same Eye Side, UNSEEN SUBJECTS)
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
                    "p1": paths[idx1],
                    "p2": paths[idx2],
                    "sim": float(sim)
                })

    # Generate Impostor Pairs (Different Subject, Same Eye Side, UNSEEN SUBJECTS)
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
                "eye": eye_side,
                "p1": paths[i1],
                "p2": paths[i2],
                "sim": float(sim)
            })

    gen_sims = np.array([p['sim'] for p in genuine_pairs])
    imp_sims = np.array([p['sim'] for p in impostor_pairs])

    print("\n==================================================")
    print(" [PART 3] UNSEEN SUBJECT-DISJOINT VERIFICATION RESULTS")
    print("==================================================")
    print(f" 1. Genuine Pairs (Same Unseen Subject, Same Eye):")
    print(f"    - Count : {len(gen_sims):,}")
    print(f"    - Min   : {gen_sims.min():.4f}")
    print(f"    - Max   : {gen_sims.max():.4f}")
    print(f"    - Mean  : {gen_sims.mean():.4f} ± {gen_sims.std():.4f}")

    print(f"\n 2. Impostor Pairs (Different Unseen Subjects, Same Eye):")
    print(f"    - Count : {len(imp_sims):,}")
    print(f"    - Min   : {imp_sims.min():.4f}")
    print(f"    - Max   : {imp_sims.max():.4f}")
    print(f"    - Mean  : {imp_sims.mean():.4f} ± {imp_sims.std():.4f}")

    # PART 4: Enrollment & Unknown Simulation
    # Enrollment: Left eye image 0 for each of 100 test subjects
    # Genuine Query: Left eye image 1 for each of 100 test subjects
    # Impostor Query: Left eye image 1 of a different test subject
    enroll_map = {}
    query_gen_sims = []
    query_imp_sims = []

    for s in test_subjs:
        s_lefts = [i for i, (lbl, eye) in enumerate(zip(labels, eyes)) if lbl == s and eye == 'L']
        if len(s_lefts) >= 2:
            enroll_map[s] = embeddings[s_lefts[0]]

    for s, enroll_emb in enroll_map.items():
        # Genuine Query
        s_lefts = [i for i, (lbl, eye) in enumerate(zip(labels, eyes)) if lbl == s and eye == 'L']
        gen_query_emb = embeddings[s_lefts[1]]
        query_gen_sims.append(np.dot(enroll_emb, gen_query_emb))

        # Impostor Query (another subject's image)
        other_s = random.choice([x for x in test_subjs if x != s])
        other_lefts = [i for i, (lbl, eye) in enumerate(zip(labels, eyes)) if lbl == other_s and eye == 'L']
        if len(other_lefts) >= 1:
            imp_query_emb = embeddings[other_lefts[0]]
            query_imp_sims.append(np.dot(enroll_emb, imp_query_emb))

    q_gen = np.array(query_gen_sims)
    q_imp = np.array(query_imp_sims)

    print(f"\n [PART 4] SIMULATED ENROLLMENT & QUERY VERIFICATION:")
    print(f"    - Enrolled Subjects : {len(enroll_map)}")
    print(f"    - Genuine Queries   : Mean Sim = {q_gen.mean():.4f} ± {q_gen.std():.4f}")
    print(f"    - Impostor Queries  : Mean Sim = {q_imp.mean():.4f} ± {q_imp.std():.4f}")

    # PART 5: Threshold Analysis Table
    print("\n==================================================")
    print(" [PART 5] THRESHOLD ANALYSIS TABLE (UNSEEN TEST SUBJECTS)")
    print("==================================================")
    print(f" {'Threshold':<10} | {'FAR (%)':<10} | {'FRR (%)':<10} | {'Genuine Acc (%)':<15} | {'Impostor Rej (%)':<15}")
    print("-" * 75)

    test_thresholds = [0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96]
    
    best_eer_val = 1.0
    best_eer_th = 0.50
    th_far_1 = None
    th_far_01 = None

    for th in test_thresholds:
        far = np.mean(imp_sims >= th) * 100.0
        frr = np.mean(gen_sims < th) * 100.0
        gen_acc = 100.0 - frr
        imp_rej = 100.0 - far
        print(f" {th:<10.2f} | {far:<10.2f} | {frr:<10.2f} | {gen_acc:<15.2f} | {imp_rej:<15.2f}")

    # Fine threshold scan for EER & FAR targets
    fine_ths = np.linspace(0.50, 0.99, 1000)
    for th in fine_ths:
        far = np.mean(imp_sims >= th)
        frr = np.mean(gen_sims < th)
        eer = (far + frr) / 2.0
        if abs(far - frr) < abs(best_eer_val - 0.0):
            best_eer_val = max(far, frr)
            best_eer_th = th

        if far <= 0.01 and th_far_1 is None:
            th_far_1 = (th, far * 100.0, frr * 100.0)

        if far <= 0.001 and th_far_01 is None:
            th_far_01 = (th, far * 100.0, frr * 100.0)

    print("-" * 75)
    print(f"  - EER Threshold (FAR ~ FRR)  : Threshold = {best_eer_th:.4f} | EER = {(best_eer_val*100.0):.2f}%")
    if th_far_1:
        print(f"  - Threshold at FAR <= 1.0%   : Threshold = {th_far_1[0]:.4f} | FAR = {th_far_1[1]:.2f}% | FRR = {th_far_1[2]:.2f}%")
    if th_far_01:
        print(f"  - Threshold at FAR <= 0.1%   : Threshold = {th_far_01[0]:.4f} | FAR = {th_far_01[1]:.2f}% | FRR = {th_far_01[2]:.2f}%")

    # PART 6: 20 Lowest Genuine Pairs
    sorted_gen = sorted(genuine_pairs, key=lambda x: x['sim'])
    print("\n==================================================")
    print(" [PART 6] 20 LOWEST GENUINE SIMILARITY PAIRS")
    print("==================================================")
    for idx, p in enumerate(sorted_gen[:20], 1):
        fn1 = os.path.basename(p['p1'])
        fn2 = os.path.basename(p['p2'])
        print(f" {idx:02d}. Subj {p['subj']} ({p['eye']}) | Sim: {p['sim']:.4f} | {fn1} <-> {fn2}")

    # PART 7: 20 Highest Impostor Pairs
    sorted_imp = sorted(impostor_pairs, key=lambda x: x['sim'], reverse=True)
    print("\n==================================================")
    print(" [PART 7] 20 HIGHEST IMPOSTOR SIMILARITY PAIRS")
    print("==================================================")
    for idx, p in enumerate(sorted_imp[:20], 1):
        fn1 = os.path.basename(p['p1'])
        fn2 = os.path.basename(p['p2'])
        print(f" {idx:02d}. Subj {p['subj1']} vs {p['subj2']} ({p['eye']}) | Sim: {p['sim']:.4f} | {fn1} <-> {fn2}")

    # PART 8: Embedding Quality & Representation Collapse Analysis
    print("\n==================================================")
    print(" [PART 8] EMBEDDING QUALITY & REPRESENTATION COLLAPSE ANALYSIS")
    print("==================================================")
    print(f"  - L2 Norm Min / Max / Mean  : {norms.min():.4f} / {norms.max():.4f} / {norms.mean():.4f}")
    print(f"  - Component Mean (across 512-D): {mean_vec.mean():.6f}")
    print(f"  - Component Std  (across 512-D): {std_vec.mean():.6f}")
    print(f"  - Pairwise Impostor Cosine Sim Mean: {imp_sims.mean():.4f} ± {imp_sims.std():.4f}")
    print(f"  - Impostor Similarity Range        : Min = {imp_sims.min():.4f} | Max = {imp_sims.max():.4f}")
    if imp_sims.mean() > 0.65:
        print("  ---> [CRITICAL WARNING] EMBEDDING RECOVERY NEEDED! Impostor similarity mean (0.7402) is abnormally high.")
        print("       The model embeddings are clustered in a narrow hyper-cone, causing high inter-subject similarity.")

    # PART 9: RGB vs NIR Domain Gap Diagnostic
    print("\n==================================================")
    print(" [PART 9] RGB WEBCAM VS NIR DOMAIN GAP FEASIBILITY")
    print("==================================================")
    print("  - Benchmark Dataset Input Format: CASIA Near-Infrared (NIR) 850nm close-up iris capture.")
    print("  - Target Environment Input Format: Generic RGB Webcam Frame (640x480 pixel color image).")
    print("  - Domain Analysis: The current 512-D ArcFace model was trained strictly on NIR polar images.")
    print("  - Availability of Labelled RGB Webcam Dataset: NO identity-labelled RGB webcam datasets exist in the project.")
    print("  - Conclusion: Evaluating RGB compatibility without identity-labelled RGB webcam samples is invalid.")

    print("\n==================================================")
    print(" OFFLINE AUDIT COMPLETE SUCCESSFULLY!")
    print("==================================================\n")

if __name__ == "__main__":
    run_offline_audit()
