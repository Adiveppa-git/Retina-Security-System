import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from ai_models.iris_recognition.train_identity_metric_model_v2 import IrisIdentityMetricCNN_v2

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

V2_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai_models/trained_models/iris_identity_metric_model_v2.pth"
)

device = torch.device("cpu")

_v2_model = None


def get_v2_model():
    global _v2_model
    if _v2_model is None:
        _v2_model = IrisIdentityMetricCNN_v2(embedding_dim=512).to(device)
        if os.path.exists(V2_MODEL_PATH):
            _v2_model.load_state_dict(torch.load(V2_MODEL_PATH, map_location=device))
            print("[V2 Embedding Pipeline] Loaded V2 model weights from:", V2_MODEL_PATH)
        else:
            print("[V2 Embedding Pipeline WARNING] V2 model weights file missing at:", V2_MODEL_PATH)
        _v2_model.eval()
    return _v2_model


# PREPROCESSING TRANSFORM EXACTLY MATCHING TRAINING PREPROCESSING
# Resizes to 64x256, converts to float Tensor [0,1], normalizes with ImageNet mean/std
v2_transform = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_v2_embedding(eye_crop_pil):
    """
    Single Source of Truth for V2 512-D L2-Normalized Feature Embedding Extraction.
    
    Input: PIL Image or RGB numpy array
    Output: 512-D L2-normalized python float list
    """
    model = get_v2_model()

    if isinstance(eye_crop_pil, Image.Image):
        pil_img = eye_crop_pil.convert("RGB")
    else:
        pil_img = Image.fromarray(eye_crop_pil).convert("RGB")

    tensor_img = v2_transform(pil_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        emb_tensor = model(tensor_img)  # 512-D L2 normalized (p=2, dim=1)

    emb_vec = emb_tensor.cpu().numpy()[0].tolist()
    return emb_vec
