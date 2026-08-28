import os
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IrisDetectionDataset(Dataset):

    def __init__(self, image_dir, label_dir):

        print("Initializing Iris Detection Dataset...")

        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = os.listdir(image_dir)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

        print("Total images loaded:", len(self.images))


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):

        img_name = self.images[idx]

        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace(".jpg",".txt"))

        image = cv2.imread(img_path)

        if image is None:
            raise Exception(f"Image not found: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)

        with open(label_path, "r") as f:
            label = f.readline().strip().split()

        label = torch.tensor([float(x) for x in label[1:]])

        return image, label