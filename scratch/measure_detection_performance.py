import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

sys.path.append(os.path.abspath("."))
from ai_models.iris_detection.model import IrisDetectorCNN
from ai_models.iris_detection.train_iris_detection import IrisDetectionDataset

def calculate_iou_xywh(box1, box2):
    # box format: [cx, cy, w, h] normalized in [0, 1]
    b1_x1 = box1[0] - box1[2] / 2.0
    b1_y1 = box1[1] - box1[3] / 2.0
    b1_x2 = box1[0] + box1[2] / 2.0
    b1_y2 = box1[1] + box1[3] / 2.0

    b2_x1 = box2[0] - box2[2] / 2.0
    b2_y1 = box2[1] - box2[3] / 2.0
    b2_x2 = box2[0] + box2[2] / 2.0
    b2_y2 = box2[1] + box2[3] / 2.0

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    b1_area = max(0.0, (b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b2_area = max(0.0, (b2_x2 - b2_x1) * (b2_y2 - b2_y1))
    union_area = b1_area + b2_area - inter_area

    if union_area <= 1e-7:
        return 0.0
    return inter_area / union_area

def evaluate_detection():
    test_img_dir = os.path.abspath("datasets/detection_dataset/test/images")
    test_lbl_dir = os.path.abspath("datasets/detection_dataset/test/labels")
    model_path = os.path.abspath("ai_models/trained_models/iris_detector.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Iris Detector on device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_ds = IrisDetectionDataset(test_img_dir, test_lbl_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    total_samples = len(test_ds)

    model = IrisDetectorCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    criterion = nn.MSELoss(reduction="none")
    all_ious = []
    all_mse = []
    
    correct_iou_05 = 0
    correct_iou_025 = 0

    with torch.no_grad():
        for imgs, bboxes in test_loader:
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)
            preds = model(imgs)
            
            mse_loss = criterion(preds, bboxes).mean(dim=1).cpu().numpy()
            all_mse.extend(mse_loss)
            
            preds_np = preds.cpu().numpy()
            bboxes_np = bboxes.cpu().numpy()
            
            for p, g in zip(preds_np, bboxes_np):
                iou = calculate_iou_xywh(p, g)
                all_ious.append(iou)
                if iou >= 0.50:
                    correct_iou_05 += 1
                if iou >= 0.25:
                    correct_iou_025 += 1

    all_ious = np.array(all_ious)
    all_mse = np.array(all_mse)

    acc_05 = (correct_iou_05 / total_samples) * 100.0
    acc_025 = (correct_iou_025 / total_samples) * 100.0
    mean_iou = float(all_ious.mean())
    median_iou = float(np.median(all_ious))
    mean_mse = float(all_mse.mean())

    print("==================================================")
    print(" IRIS DETECTION MODEL ACTUAL TEST EVALUATION")
    print("==================================================")
    print(f"Total Test Samples Evaluated: {total_samples:,}")
    print(f"Mean Bounding Box MSE Loss  : {mean_mse:.6f}")
    print(f"Mean Intersection-over-Union (mIoU): {mean_iou:.4f} ({mean_iou*100:.2f}%)")
    print(f"Median IoU                  : {median_iou:.4f} ({median_iou*100:.2f}%)")
    print(f"Detection Accuracy (IoU >= 0.50)   : {acc_05:.2f}% ({correct_iou_05:,}/{total_samples:,})")
    print(f"Detection Accuracy (IoU >= 0.25)   : {acc_025:.2f}% ({correct_iou_025:,}/{total_samples:,})")
    print("==================================================")

if __name__ == "__main__":
    evaluate_detection()
