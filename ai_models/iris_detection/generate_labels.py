import os
import cv2
import numpy as np

print("Starting automatic label generation...")

train_img_dir = "datasets/detection_dataset/train/images"
train_label_dir = "datasets/detection_dataset/train/labels"

test_img_dir = "datasets/detection_dataset/test/images"
test_label_dir = "datasets/detection_dataset/test/labels"

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(test_label_dir, exist_ok=True)


def process_folder(img_dir, label_dir):

    images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png'))]

    for i, img_name in enumerate(images):

        img_path = os.path.join(img_dir, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9,9), 2)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=80
        )

        if circles is None:
            continue

        circles = np.uint16(np.around(circles))
        x,y,r = circles[0][0]

        h, w = image.shape[:2]

        x_center = x / w
        y_center = y / h
        width = (2*r) / w
        height = (2*r) / h

        label_path = os.path.join(label_dir, img_name.split(".")[0] + ".txt")

        with open(label_path, "w") as f:
            f.write(f"0 {x_center} {y_center} {width} {height}")

        if i % 1000 == 0:
            print(f"Processed {i} images")



print("Generating test labels...")
process_folder(test_img_dir, test_label_dir)

print("Label generation finished")