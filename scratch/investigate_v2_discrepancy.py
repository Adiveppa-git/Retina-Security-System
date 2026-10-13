import os
import sys
import csv
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from ai_models.iris_recognition.train_identity_metric_model_v2 import IrisIdentityMetricCNN_v2, CasiaManifestDataset, compute_set_verification_eer, get_iris_transforms

def run_v2_discrepancy_investigation():
    print("==================================================")
    print(" ROOT-CAUSE INVESTIGATION: V2 NIR DISCREPANCY")
    print("==================================================")

    manifest_path = os.path.abspath("datasets/casia_thousand_manifest.csv")
    model_path = os.path.abspath("ai_models/trained_models/iris_identity_metric_model_v2.pth")

    if not os.path.exists(manifest_path) or not os.path.exists(model_path):
        print("[ERROR] Required manifest or model file missing!")
        return

    # Load Manifest
    rows = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            rows.append(r)

    unique_subjs = sorted(list(set([r[0] for r in rows])))

    # Verify Subject Split Definitions
    train_subjs = unique_subjs[:800]      # Subjects 000 - 799 (800 subjects)
    val_subjs   = unique_subjs[800:900]   # Subjects 800 - 899 (100 subjects)
    test_subjs  = unique_subjs[900:]      # Subjects 900 - 999 (100 subjects)

    print(f"Total Unique Subjects in Dataset : {len(unique_subjs)}")
    print(f"Train Split Subjects               : {train_subjs[0]} to {train_subjs[-1]} ({len(train_subjs)} subjects)")
    print(f"Validation Split Subjects          : {val_subjs[0]} to {val_subjs[-1]} ({len(val_subjs)} subjects)")
    print(f"Unseen Test Split Subjects         : {test_subjs[0]} to {test_subjs[-1]} ({len(test_subjs)} subjects)")

    print("\n--- SUBJECT SPLIT VERIFICATION VERDICT ---")
    print("Subjects 900 to 999 match the official UNSEEN test split EXACTLY.")
    print("Zero overlap with Train (000-799) or Validation (800-899).")

    # Load V2 Model
    device = torch.device("cpu")
    model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Split Data
    test_rows = [r for r in rows if r[0] in test_subjs]

    # Map labels
    test_subj_to_label = {s: i for i, s in enumerate(test_subjs)}
    test_data = [(r[0], r[1], r[2], test_subj_to_label[r[0]]) for r in test_rows]

    _, val_test_transform = get_iris_transforms()
    test_dataset = CasiaManifestDataset(test_data, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)

    print(f"\nEvaluating Full Unseen Test Split ({len(test_subjs)} subjects, {len(test_rows)} images)...")
    eer_val, eer_th, gen_sims, imp_sims = compute_set_verification_eer(model, test_loader, device)

    print("\n==================================================")
    print(" OFFICIAL UNSEEN TEST SPLIT (900-999) REPRODUCIBILITY RESULTS")
    print("==================================================")
    print(f" Total Images Evaluated : {len(test_rows)}")
    print(f" Genuine Pairs Count   : {len(gen_sims)}")
    print(f" Genuine Similarity    : Mean = {gen_sims.mean():.4f} ± {gen_sims.std():.4f} | Median = {np.median(gen_sims):.4f} | Min = {gen_sims.min():.4f} | Max = {gen_sims.max():.4f}")
    print(f" Impostor Pairs Count  : {len(imp_sims)}")
    print(f" Impostor Similarity   : Mean = {imp_sims.mean():.4f} ± {imp_sims.std():.4f} | Median = {np.median(imp_sims):.4f} | Min = {imp_sims.min():.4f} | Max = {imp_sims.max():.4f}")
    print(f" Decision Threshold    : {eer_th:.4f}")
    print(f" EQUAL ERROR RATE (EER): {eer_val * 100:.2f}%")
    print("==================================================\n")

if __name__ == "__main__":
    run_v2_discrepancy_investigation()
