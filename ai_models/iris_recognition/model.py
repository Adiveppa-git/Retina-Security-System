import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisRecognitionCNN(nn.Module):
    def __init__(self):
        super(IrisRecognitionCNN, self).__init__()

        print("Initializing Feature Extractor Model...")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Linear(512, 128)  # 🔥 128-D feature vector
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)

        # normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)

        return x