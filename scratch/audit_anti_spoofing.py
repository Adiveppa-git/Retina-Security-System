import os
import sys
import hashlib
import csv
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))
from ai_models.anti_spoofing.model import AntiSpoofCNN

def file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def audit_antispoof():
    base_dir = os.path.abspath("datasets/anti_spoof_dataset")
    train_real_dir = os.path.join(base_dir, "train", "real")
    train_fake_dir = os.path.join(base_dir, "train", "fake")
    test_real_dir = os.path.join(base_dir, "test", "real")
    test_fake_dir = os.path.join(base_dir, "test", "fake")

    print("==================================================")
    print(" 1. HASH DUPLICATION & DATA LEAKAGE AUDIT")
    print("==================================================")
    train_hashes = set()
    train_count = 0
    for d in [train_real_dir, train_fake_dir]:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                train_hashes.add(file_hash(p))
                train_count += 1
    print(f"Total Train Images: {train_count:,} (Unique Hashes: {len(train_hashes):,})")

    overlap_count = 0
    test_count = 0
    for d in [test_real_dir, test_fake_dir]:
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                test_count += 1
                if file_hash(p) in train_hashes:
                    overlap_count += 1
    print(f"Total Test Images : {test_count:,}")
    print(f"Exact Duplicate Hash Overlap Between Train & Test: {overlap_count}")
    if overlap_count == 0:
        print("[VERIFIED] Zero image-level data leakage between training and testing splits.")
    else:
        print(f"[WARNING] Detected {overlap_count} overlapping images!")

    print("\n==================================================")
    print(" 2. DIRECT PREDICTION & CONFUSION MATRIX AUDIT")
    print("==================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.abspath("ai_models/trained_models/anti_spoof_model.pth")
    test_dir = os.path.join(base_dir, "test")

    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = ImageFolder(test_dir, transform=transform)
    fake_idx = test_ds.class_to_idx["fake"] # 0
    real_idx = test_ds.class_to_idx["real"] # 1

    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    model = AntiSpoofCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    output_csv = os.path.abspath("scratch/antispoof_verified_predictions.csv")
    records = []

    tp, fn, tn, fp = 0, 0, 0, 0

    idx = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, 1)

            for p_tensor, l_tensor, prob_tensor in zip(preds, labels, probs):
                img_path, _ = test_ds.samples[idx]
                gt = "real" if l_tensor.item() == real_idx else "fake"
                pred = "real" if p_tensor.item() == real_idx else "fake"
                conf = float(prob_tensor[p_tensor.item()].item())

                if gt == "real":
                    if pred == "real":
                        tp += 1
                        status = "CORRECT_TP"
                    else:
                        fn += 1
                        status = "MISCLASSIFIED_FN"
                else:
                    if pred == "fake":
                        tn += 1
                        status = "CORRECT_TN"
                    else:
                        fp += 1
                        status = "MISCLASSIFIED_FP"

                records.append([os.path.basename(img_path), gt, pred, f"{conf*100:.2f}%", status])
                idx += 1

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Ground_Truth", "Predicted", "Confidence", "Verification_Status"])
        writer.writerows(records)

    total = tp + fn + tn + fp
    total_real = tp + fn
    total_fake = tn + fp
    acc = (tp + tn) / total * 100.0
    bpcer = (fn / total_real) * 100.0
    apcer = (fp / total_fake) * 100.0
    acer = (bpcer + apcer) / 2.0

    print(f"Total Evaluated Test Samples: {total}")
    print(f"Confusion Matrix:")
    print(f"  TP: {tp} (Real as Real)")
    print(f"  FN: {fn} (Real as Fake - BPCER numerator)")
    print(f"  TN: {tn} (Fake as Fake)")
    print(f"  FP: {fp} (Fake as Real - APCER numerator)")
    print(f"Accuracy : {acc:.4f}% ({tp+tn}/{total})")
    print(f"BPCER    : {bpcer:.4f}% ({fn}/{total_real})")
    print(f"APCER    : {apcer:.4f}% ({fp}/{total_fake})")
    print(f"ACER     : {acer:.4f}%")
    print(f"Detailed predictions saved to: {output_csv}")

if __name__ == "__main__":
    audit_antispoof()
