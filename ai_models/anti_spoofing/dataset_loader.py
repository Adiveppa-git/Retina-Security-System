import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class IrisSpoofDataset(Dataset):

    def __init__(self, real_dir, fake_dir):

        print("Initializing dataset loader...")

        self.image_paths = []
        self.labels = []

        real_count = 0
        fake_count = 0

        for file in os.listdir(real_dir):
            if file.endswith(".jpg"):
                self.image_paths.append(os.path.join(real_dir, file))
                self.labels.append(1)
                real_count += 1

        for file in os.listdir(fake_dir):
            if file.endswith(".jpg"):
                self.image_paths.append(os.path.join(fake_dir, file))
                self.labels.append(0)
                fake_count += 1

        print(f"Real images loaded: {real_count}")
        print(f"Fake images loaded: {fake_count}")

        self.transform = transforms.Compose([
            transforms.Resize((48,48)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label