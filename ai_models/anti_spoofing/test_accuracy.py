import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from PIL import Image
import torchvision.transforms as transforms

from ai_models.anti_spoofing.model import AntiSpoofCNN


print("Loading trained model...")

model = AntiSpoofCNN()
model.load_state_dict(torch.load("ai_models/trained_models/anti_spoof_model.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully")


transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor()
])


real_dir = "datasets/anti_spoof_dataset/test/real"
fake_dir = "datasets/anti_spoof_dataset/test/fake"

real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(".jpg")]
fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.endswith(".jpg")]

print("Real images:", len(real_images))
print("Fake images:", len(fake_images))


correct = 0
total = 0


print("\nTesting REAL images...")

for i, img_path in enumerate(real_images):

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 1:
        correct += 1

    total += 1

    if i % 2000 == 0:
        print(f"Processed REAL images: {i}/{len(real_images)}")


print("\nTesting FAKE images...")

for i, img_path in enumerate(fake_images):

    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    if predicted.item() == 0:
        correct += 1

    total += 1

    if i % 2000 == 0:
        print(f"Processed FAKE images: {i}/{len(fake_images)}")


accuracy = (correct / total) * 100

print("\n-------------------------")
print("Total images tested:", total)
print("Correct predictions:", correct)
print(f"Final Accuracy: {accuracy:.2f}%")
print("-------------------------")