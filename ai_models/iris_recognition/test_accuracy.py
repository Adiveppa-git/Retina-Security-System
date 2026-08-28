import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from ai_models.iris_recognition.model import IrisRecognitionCNN


print("Loading iris recognition test dataset...")

# image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])

# load test dataset
test_dataset = datasets.ImageFolder(
    "datasets/recognition_dataset/test",
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

num_classes = len(test_dataset.classes)

print("Test images:", len(test_dataset))
print("Persons in test dataset:", num_classes)


print("\nLoading trained model...")

# ✅ FIX: use correct class count
model = IrisRecognitionCNN(num_classes)

model.load_state_dict(
    torch.load(
        "ai_models/trained_models/iris_recognition_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully")


correct = 0
total = 0


print("\nTesting recognition...")

for i, (image, label) in enumerate(test_loader):

    with torch.no_grad():

        output = model(image)

        _, predicted = torch.max(output, 1)

    if predicted.item() == label.item():
        correct += 1

    total += 1

    if i % 1000 == 0:
        print(f"Processed images: {i}/{len(test_dataset)}")


accuracy = (correct / total) * 100


print("\n---------------------------")
print("Total images tested:", total)
print("Correct predictions:", correct)
print(f"Recognition Accuracy: {accuracy:.2f}%")
print("---------------------------")