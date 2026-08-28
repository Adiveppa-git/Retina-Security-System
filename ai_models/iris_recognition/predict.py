import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from PIL import Image
import torchvision.transforms as transforms
from torchvision import datasets

from ai_models.iris_recognition.model import IrisRecognitionCNN


# check argument
if len(sys.argv) != 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()


image_path = sys.argv[1]


print("Loading trained iris recognition model...")

# number of classes
dataset = datasets.ImageFolder("datasets/recognition_dataset/train")
num_classes = len(dataset.classes)

model = IrisRecognitionCNN(num_classes)
model.load_state_dict(
    torch.load(
        "ai_models/trained_models/iris_recognition_model.pth",
        map_location="cpu"
    )
)

model.eval()

print("Model loaded successfully")


# dataset mapping
dataset = datasets.ImageFolder("datasets/recognition_dataset/train")

idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}


# image transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


print("\nLoading image:", image_path)

image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)


# prediction
with torch.no_grad():

    output = model(image)

    _, predicted = torch.max(output, 1)


predicted_index = predicted.item()

predicted_folder = idx_to_class[predicted_index]


print("\n---------------------------")
print("Predicted Class Index :", predicted_index)
print("Predicted Folder     :", predicted_folder)
print("---------------------------")