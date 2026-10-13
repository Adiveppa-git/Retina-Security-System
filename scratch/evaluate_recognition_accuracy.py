import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from ai_models.iris_recognition.model import IrisRecognitionCNN


def calculate_recognition_test_accuracy():
    print("==================================================")
    print(" CALCULATING IRIS RECOGNITION TEST ACCURACY...")
    print("==================================================")

    test_dir = os.path.abspath("datasets/recognition_dataset/test")

    if not os.path.exists(test_dir):
        print("Error: Test directory not found!")
        return

    device = torch.device("cpu")
    model_path = os.path.abspath("ai_models/trained_models/iris_feature_model.pth")
    model = IrisRecognitionCNN().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Filter out any non-class items
    test_dataset = ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"  Test Images Loaded : {len(test_dataset)} across {len(test_dataset.classes)} subject classes")

    correct_matches = 0
    total_comparisons = 0
    tp, fp, tn, fn = 0, 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            embeddings = model(images)
            n = len(embeddings)

            for i in range(n):
                for j in range(i + 1, min(n, i + 10)):
                    sim = torch.nn.functional.cosine_similarity(
                        embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                    ).item()

                    is_same_subject = (labels[i] == labels[j]).item()
                    total_comparisons += 1

                    if is_same_subject:
                        if sim >= 0.75:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if sim < 0.75:
                            tn += 1
                        else:
                            fp += 1

    accuracy = (tp + tn) / max(1, total_comparisons) * 100
    precision = tp / max(1, tp + fp) * 100
    recall = tp / max(1, tp + fn) * 100
    f1 = 2 * (precision * recall) / max(1e-5, precision + recall)

    print(f"\n==================================================")
    print(f" IRIS RECOGNITION TEST ACCURACY & CONFUSION MATRIX")
    print(f"==================================================")
    print(f"                 PREDICTED SAME    PREDICTED DIFFERENT")
    print(f"  ACTUAL SAME   |  TP = {tp:<9} |  FN = {fn:<9} |")
    print(f"  ACTUAL DIFF   |  FP = {fp:<9} |  TN = {tn:<9} |")
    print(f"--------------------------------------------------")
    print(f"  Total Comparisons Evaluated : {total_comparisons}")
    print(f"  Test Accuracy               : {accuracy:.2f}%")
    print(f"  Precision                   : {precision:.2f}%")
    print(f"  Recall                      : {recall:.2f}%")
    print(f"  F1-Score                    : {f1:.2f}%")
    print(f"==================================================\n")


if __name__ == "__main__":
    calculate_recognition_test_accuracy()
