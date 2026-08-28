import os
import shutil
import random

base = "datasets/recognition_dataset"

sources = [
    "CASIA-Iris-Distance",
    "CASIA-Iris-Lamp",
    "CASIA-Iris-Thousand",
    "IITD_database"
]

train_dir = os.path.join(base, "train")
test_dir = os.path.join(base, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

person_index = 0

for dataset in sources:

    dataset_path = os.path.join(base, dataset)

    print("Processing:", dataset)

    for person in os.listdir(dataset_path):

        person_path = os.path.join(dataset_path, person)

        if not os.path.isdir(person_path):
            continue

        images = []

        # scan all images inside folder
        for root, dirs, files in os.walk(person_path):
            for f in files:
                if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff")):
                    images.append(os.path.join(root,f))

        if len(images) == 0:
            continue

        random.shuffle(images)

        split = int(len(images) * 0.8)

        train_imgs = images[:split]
        test_imgs = images[split:]

        person_id = f"{person_index:04d}"

        train_person = os.path.join(train_dir, person_id)
        test_person = os.path.join(test_dir, person_id)

        os.makedirs(train_person, exist_ok=True)
        os.makedirs(test_person, exist_ok=True)

        for img in train_imgs:
            shutil.copy(img, os.path.join(train_person, os.path.basename(img)))

        for img in test_imgs:
            shutil.copy(img, os.path.join(test_person, os.path.basename(img)))

        person_index += 1

print("All datasets merged successfully")