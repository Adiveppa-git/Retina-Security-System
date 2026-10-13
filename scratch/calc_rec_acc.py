import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from ai_models.iris_recognition.model import IrisRecognitionCNN

test_dir = os.path.abspath("datasets/recognition_dataset/test")
model_path = os.path.abspath("ai_models/trained_models/iris_feature_model.pth")

device = torch.device("cpu")
model = IrisRecognitionCNN().to(device)

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(test_dir, transform=transform)
loader = DataLoader(dataset, batch_size=100, shuffle=False)

for images, labels in loader:
    embeddings = model(images)
    tp, fp, tn, fn = 0, 0, 0, 0
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, min(n, i + 10)):
            sim = torch.nn.functional.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            same = (labels[i] == labels[j]).item()
            if same:
                if sim >= 0.75:
                    tp += 1
                else:
                    fn += 1
            else:
                if sim < 0.75:
                    tn += 1
                else:
                    fp += 1

    total = tp + fp + tn + fn
    acc = (tp + tn) / max(1, total) * 100
    prec = tp / max(1, tp + fp) * 100
    rec = tp / max(1, tp + fn) * 100
    f1 = 2 * (prec * rec) / max(1e-5, prec + rec)

    print(f"==================================================")
    print(f" IRIS RECOGNITION TEST ACCURACY REPORT")
    print(f"==================================================")
    print(f"  Test Accuracy   : {acc:.2f}%")
    print(f"  Precision       : {prec:.2f}%")
    print(f"  Recall          : {rec:.2f}%")
    print(f"  F1-Score        : {f1:.2f}%")
    print(f"  Confusion Matrix: TP={tp} | FP={fp} | TN={tn} | FN={fn}")
    print(f"==================================================")
    break
