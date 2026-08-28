import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import cv2
import torchvision.transforms as transforms

from ai_models.iris_detection.model import IrisDetectorCNN


def predict(image_path):

    print("Loading iris detection model...")

    model = IrisDetectorCNN()
    model.load_state_dict(torch.load("ai_models/trained_models/iris_detector.pth", map_location="cpu"))
    model.eval()

    print("Model loaded successfully")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        return

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # -----------------------------

    h, w = image.shape[:2]

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)

    x, y, bw, bh = output[0]

    x = int(x.item() * w)
    y = int(y.item() * h)
    bw = int(bw.item() * w)
    bh = int(bh.item() * h)

    x1 = int(x - bw/2)
    y1 = int(y - bh/2)
    x2 = int(x + bw/2)
    y2 = int(y + bh/2)

    # draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 3)

    # draw center point
    cx = int(x)
    cy = int(y)
    cv2.circle(image, (cx,cy), 5, (0,0,255), -1)

    print("Prediction complete")

    # resize for easier viewing
    display = cv2.resize(image, (900,500))

    cv2.imshow("Iris Detection", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    predict(sys.argv[1])