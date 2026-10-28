import os
import sys
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))
from ai_models.anti_spoofing.model import AntiSpoofCNN

def evaluate_antispoof():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Anti-Spoofing on device: {device}")
    
    test_dir = os.path.abspath("datasets/anti_spoof_dataset/test")
    model_path = os.path.abspath("ai_models/trained_models/anti_spoof_model.pth")
    
    transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_ds = ImageFolder(test_dir, transform=transform)
    print("Class mapping:", test_ds.class_to_idx)
    # fake=0, real=1
    fake_idx = test_ds.class_to_idx.get("fake", 0)
    real_idx = test_ds.class_to_idx.get("real", 1)
    
    loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    
    model = AntiSpoofCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    tp = 0 # real predicted as real
    fn = 0 # real predicted as fake (BPCER)
    tn = 0 # fake predicted as fake
    fp = 0 # fake predicted as real (APCER)
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            for p, l in zip(preds, labels):
                if l.item() == real_idx: # Bona fide / Real
                    if p.item() == real_idx:
                        tp += 1
                    else:
                        fn += 1
                else: # Attack / Fake / Spoof
                    if p.item() == fake_idx:
                        tn += 1
                    else:
                        fp += 1
                        
    total = tp + fn + tn + fp
    total_real = tp + fn
    total_fake = tn + fp
    accuracy = (tp + tn) / total * 100.0
    bpcer = (fn / total_real) * 100.0 if total_real > 0 else 0.0 # Bona Fide Presentation Classification Error Rate
    apcer = (fp / total_fake) * 100.0 if total_fake > 0 else 0.0 # Attack Presentation Classification Error Rate
    acer = (bpcer + apcer) / 2.0 # Average Classification Error Rate
    precision = tp / (tp + fp) * 100.0 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("==================================================")
    print(" ANTI-SPOOFING MODEL ACTUAL TEST EVALUATION")
    print("==================================================")
    print(f"Total Test Samples Evaluated: {total:,}")
    print(f"  - Bona Fide (Real Human Eye) Samples: {total_real:,}")
    print(f"  - Presentation Attack (Spoof) Samples: {total_fake:,}")
    print("--------------------------------------------------")
    print(f"Confusion Matrix:")
    print(f"  TP (Real correctly identified as Real): {tp:,}")
    print(f"  FN (Real misclassified as Spoof)     : {fn:,}")
    print(f"  TN (Spoof correctly identified as Spoof): {tn:,}")
    print(f"  FP (Spoof misclassified as Real)     : {fp:,}")
    print("--------------------------------------------------")
    print(f"Overall Test Accuracy: {accuracy:.4f}% ({tp+tn}/{total})")
    print(f"BPCER (Bona Fide Presentation Error Rate / False Reject of Live): {bpcer:.4f}% ({fn}/{total_real})")
    print(f"APCER (Attack Presentation Error Rate / False Accept of Spoof): {apcer:.4f}% ({fp}/{total_fake})")
    print(f"ACER (Average Classification Error Rate): {acer:.4f}%")
    print(f"Precision: {precision:.4f}%")
    print(f"Recall: {recall:.4f}%")
    print(f"F1-Score: {f1:.4f}%")
    print("==================================================")

if __name__ == "__main__":
    evaluate_antispoof()
