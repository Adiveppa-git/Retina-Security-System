import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import IrisRecognitionCNN


print("Loading dataset...")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    "datasets/recognition_dataset/train",
    transform=transform
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("Total images:", len(dataset))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IrisRecognitionCNN()
model.load_state_dict(
    torch.load("ai_models/trained_models/iris_feature_model.pth", map_location=device)
)
model.eval()
model.to(device)
model.eval()


features_db = []
labels_db = []


print("Extracting features...")

with torch.no_grad():
    for img, label in loader:
        img = img.to(device)

        feature = model(img)

        features_db.append(feature.cpu())
        labels_db.append(label)


features_db = torch.cat(features_db)
labels_db = torch.tensor(labels_db)


torch.save({
    "features": features_db,
    "labels": labels_db,
    "classes": dataset.classes
}, "ai_models/trained_models/iris_features.pth")


print("✅ Features saved successfully!")