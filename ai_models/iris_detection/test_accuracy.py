import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
from PIL import Image
import torchvision.transforms as transforms

from ai_models.iris_detection.model import IrisDetectorCNN


print("Loading iris detection model...")

model = IrisDetectorCNN()
model.load_state_dict(torch.load("ai_models/trained_models/iris_detector.pth", map_location="cpu"))
model.eval()

print("Model loaded successfully")


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


IMAGE_DIR = "datasets/detection_dataset/test/images"
LABEL_DIR = "datasets/detection_dataset/test/labels"


total = 0
correct = 0

IOU_THRESHOLD = 0.5


def calculate_iou(box1, box2):

    x1,y1,x2,y2 = box1
    x1g,y1g,x2g,y2g = box2

    xi1 = max(x1,x1g)
    yi1 = max(y1,y1g)
    xi2 = min(x2,x2g)
    yi2 = min(y2,y2g)

    inter = max(0,xi2-xi1) * max(0,yi2-yi1)

    area1 = (x2-x1)*(y2-y1)
    area2 = (x2g-x1g)*(y2g-y1g)

    union = area1 + area2 - inter

    if union == 0:
        return 0

    return inter/union


print("\nTesting images...\n")


images = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg",".jpeg",".png"))]


for i,img_name in enumerate(images):

    img_path = os.path.join(IMAGE_DIR,img_name)
    label_path = os.path.join(LABEL_DIR,img_name.split(".")[0] + ".txt")

    if not os.path.exists(label_path):
        continue

    image = Image.open(img_path).convert("RGB")

    w,h = image.size

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)

    px,py,pw,ph = output[0]

    px = px.item()
    py = py.item()
    pw = pw.item()
    ph = ph.item()


    pred_box = [
        int((px - pw/2) * w),
        int((py - ph/2) * h),
        int((px + pw/2) * w),
        int((py + ph/2) * h)
    ]


    with open(label_path,"r") as f:
        line = f.readline().split()

    _,gx,gy,gw,gh = map(float,line)


    gt_box = [
        int((gx - gw/2) * w),
        int((gy - gh/2) * h),
        int((gx + gw/2) * w),
        int((gy + gh/2) * h)
    ]


    iou = calculate_iou(pred_box,gt_box)

    if iou >= IOU_THRESHOLD:
        correct += 1

    total += 1

    if i % 1000 == 0:
        print(f"Processed images: {i}/{len(images)}")


accuracy = (correct/total) * 100


print("\n-------------------------")
print("Total images tested:", total)
print("Correct detections:", correct)
print(f"Detection Accuracy: {accuracy:.2f}%")
print("-------------------------")