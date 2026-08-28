import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from PIL import Image
import torchvision.transforms as transforms

from ai_models.anti_spoofing.model import AntiSpoofCNN


def load_model():

    print("Loading trained anti-spoof model...")

    model = AntiSpoofCNN()

    model_path = "ai_models/trained_models/anti_spoof_model.pth"

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()

    print("Model loaded successfully")

    return model


def preprocess_image(image_path):

    print("Loading image for prediction...")

    transform = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")

    image = transform(image)

    image = image.unsqueeze(0)

    return image


def predict(image_path):

    model = load_model()

    image = preprocess_image(image_path)

    print("Running anti-spoof prediction...")

    with torch.no_grad():

        outputs = model(image)

        _, predicted = torch.max(outputs, 1)

        if predicted.item() == 1:
            print("Prediction Result: REAL IRIS")
        else:
            print("Prediction Result: FAKE IRIS")


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        exit()

    image_path = sys.argv[1]

    predict(image_path)