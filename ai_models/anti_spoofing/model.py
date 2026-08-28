import torch
import torch.nn as nn
import torch.nn.functional as F


class AntiSpoofCNN(nn.Module):

    def __init__(self):
        super(AntiSpoofCNN, self).__init__()

        print("Initializing CNN model...")

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # Original FC layers
        self.fc1 = nn.Linear(64 * 6 * 6, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        print("Shape before flatten:", x.shape)

        x = x.view(x.size(0), -1)

        print("Shape before fc1:", x.shape)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x