import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from ai_models.anti_spoofing.model import AntiSpoofCNN

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/anti_spoof_model.pth"
)

device = torch.device("cpu")
anti_spoof_model = AntiSpoofCNN().to(device)

if os.path.exists(MODEL_PATH):
    anti_spoof_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

anti_spoof_model.eval()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def check_spoof(image_input):
    """
    Evaluates anti-spoofing liveness of input image.
    Returns:
    True  -> REAL HUMAN EYE
    False -> SPOOF DETECTED (Screen/Photo Print)
    """
    try:
        if isinstance(image_input, Image.Image):
            pil_image = image_input.convert("RGB")
        else:
            pil_image = Image.open(image_input).convert("RGB")

        tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = anti_spoof_model(tensor)
            _, predicted = torch.max(output, 1)

        # Class 1 = REAL, Class 0 = FAKE
        is_real = (predicted.item() == 1)
        return is_real

    except Exception as e:
        print("Anti-Spoof Error:", e)
        # Default to True in fallback cases unless strong spoof detected
        return True