import os
import shutil
import random

src = "datasets/recognition_dataset/CASIA-Iris-Distance"
train_dir = "datasets/recognition_dataset/train"
test_dir = "datasets/recognition_dataset/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for person in os.listdir(src):

    person_path = os.path.join(src, person)

    if not os.path.isdir(person_path):
        continue

    images = os.listdir(person_path)

    random.shuffle(images)

    split = int(len(images) * 0.8)

    train_imgs = images[:split]
    test_imgs = images[split:]

    os.makedirs(os.path.join(train_dir, person), exist_ok=True)
    os.makedirs(os.path.join(test_dir, person), exist_ok=True)

    for img in train_imgs:
        shutil.copy(
            os.path.join(person_path, img),
            os.path.join(train_dir, person, img)
        )

    for img in test_imgs:
        shutil.copy(
            os.path.join(person_path, img),
            os.path.join(test_dir, person, img)
        )

print("Dataset split completed successfully")