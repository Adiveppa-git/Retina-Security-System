import torch
import torch.nn.functional as F
import os
from PIL import Image
import torchvision.transforms as transforms

from ai_models.iris_recognition.model import IrisRecognitionCNN

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

USER_DB_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/user_features.pth"
)

device = torch.device("cpu")

print("Initializing Recognition Model...")

model = IrisRecognitionCNN().to(device)

model.load_state_dict(
    torch.load(
        os.path.join(
            BASE_DIR,
            "ai_models/trained_models/iris_feature_model.pth"
        ),
        map_location=device
    )
)

model.eval()


def recognize_iris(image):

    try:

        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = model(image)

        feature = F.normalize(feature, dim=1)

        if not os.path.exists(USER_DB_PATH):
            return {
                "person": "No Users Registered",
                "similarity": 0.0
            }

        db = torch.load(USER_DB_PATH)

        if len(db["names"]) == 0:
            return {
                "person": "No Users Registered",
                "similarity": 0.0
            }

        best_score = -1
        best_user = "Unknown"

        for stored_feature, name in zip(
            db["features"],
            db["names"]
        ):

            stored_feature = stored_feature.unsqueeze(0)

            score = F.cosine_similarity(
                feature.cpu(),
                stored_feature,
                dim=1
            ).item()

            if score > best_score:
                best_score = score
                best_user = name

        print("================================")
        print("Best Match:", best_user)
        print("Similarity Score:", best_score)
        print("Threshold:", 0.80)
        print("================================")

        if best_score > 0.80:
            return {
                "person": best_user,
                "similarity": round(best_score, 4)
            }

        return {
            "person": "Unknown",
            "similarity": round(best_score, 4)
        }

    except Exception as e:
        print("Recognition Error:", e)

        return {
            "person": "Error",
            "similarity": 0.0
        }