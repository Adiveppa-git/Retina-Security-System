import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from ai_models.iris_recognition.model import IrisRecognitionCNN

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../")
)

USER_DB_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/user_features.pth"
)

# =====================================================
# Device
# =====================================================

device = torch.device("cpu")

# =====================================================
# Load Recognition Model
# =====================================================

print("Initializing Feature Extractor Model...")

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

# =====================================================
# Register User
# =====================================================

def register_user(name, image):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor()
    ])

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feature = model(image)

    feature = F.normalize(feature, dim=1)

    if os.path.exists(USER_DB_PATH):

        try:
            db = torch.load(USER_DB_PATH)

        except:
            db = {
                "features": [],
                "names": []
            }

    else:

        db = {
            "features": [],
            "names": []
        }

    db["features"].append(feature.squeeze(0).cpu())
    db["names"].append(name)

    torch.save(db, USER_DB_PATH)

    return {
        "status": "success",
        "user": name,
        "total_users": len(db["names"])
    }


# =====================================================
# Get All Users
# =====================================================

def get_all_users():

    if not os.path.exists(USER_DB_PATH):
        return []

    db = torch.load(USER_DB_PATH)

    users = []

    for i, name in enumerate(db["names"], start=1):

        users.append({

            "id": i,

            "name": name

        })

    return users


# =====================================================
# Delete User
# =====================================================

def delete_user(user_name):

    if not os.path.exists(USER_DB_PATH):

        return {

            "status": "error",

            "message": "Database not found"

        }

    db = torch.load(USER_DB_PATH)

    if user_name not in db["names"]:

        return {

            "status": "error",

            "message": "User not found"

        }

    index = db["names"].index(user_name)

    db["names"].pop(index)

    db["features"].pop(index)

    torch.save(db, USER_DB_PATH)

    return {

        "status": "success",

        "message": f"{user_name} deleted successfully",

        "total_users": len(db["names"])

    }


# =====================================================
# Update User ID
# =====================================================

def update_user(old_name, new_name):

    if not os.path.exists(USER_DB_PATH):

        return {

            "status": "error",

            "message": "Database not found"

        }

    db = torch.load(USER_DB_PATH)

    if old_name not in db["names"]:

        return {

            "status": "error",

            "message": "User not found"

        }

    if new_name in db["names"]:

        return {

            "status": "error",

            "message": "User ID already exists"

        }

    index = db["names"].index(old_name)

    db["names"][index] = new_name

    torch.save(db, USER_DB_PATH)

    return {

        "status": "success",

        "message": f"{old_name} updated to {new_name}"

    }