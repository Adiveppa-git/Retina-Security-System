import os
import torch
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from ai_models.iris_detection.model import IrisDetectorCNN

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/iris_detector.pth"
)

device = torch.device("cpu")

print("Loading Iris Detection Model...")

model = IrisDetectorCNN().to(device)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def detect_iris(image):

    try:

        if isinstance(image, Image.Image):
            rgb = image
        else:
            rgb = Image.open(image).convert("RGB")

        original = np.array(rgb)

        h, w = original.shape[:2]

        tensor = transform(rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(tensor)

        x, y, bw, bh = output[0]

        x = int(x.item() * w)
        y = int(y.item() * h)
        bw = int(bw.item() * w)
        bh = int(bh.item() * h)

        x1 = max(0, x - bw // 2)
        y1 = max(0, y - bh // 2)
        x2 = min(w, x + bw // 2)
        y2 = min(h, y + bh // 2)

        cropped = original[y1:y2, x1:x2]

        if cropped.size == 0:
            return rgb

        return Image.fromarray(cropped)

    except Exception as e:

        print("Detection Error:", e)

        return image