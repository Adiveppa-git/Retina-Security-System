import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class TripletDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])

        self.classes = os.listdir(root_dir)

        self.image_paths = {}

        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            images = os.listdir(cls_path)
            if len(images) >= 2:
                self.image_paths[cls] = [
                    os.path.join(cls_path, img) for img in images
                ]

        self.classes = list(self.image_paths.keys())

    def __len__(self):
        return 10000  # random sampling

    def __getitem__(self, idx):
        anchor_class = random.choice(self.classes)

        positive_class = anchor_class
        negative_class = random.choice(self.classes)

        while negative_class == anchor_class:
            negative_class = random.choice(self.classes)

        anchor_img = random.choice(self.image_paths[anchor_class])
        positive_img = random.choice(self.image_paths[positive_class])
        negative_img = random.choice(self.image_paths[negative_class])

        anchor = self.transform(Image.open(anchor_img))
        positive = self.transform(Image.open(positive_img))
        negative = self.transform(Image.open(negative_img))

        return anchor, positive, negative