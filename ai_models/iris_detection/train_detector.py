import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
torch.set_num_threads(4)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ai_models.iris_detection.dataset_loader import IrisDetectionDataset
from ai_models.iris_detection.model import IrisDetectorCNN


def train():

    print("Starting Iris Detection Training System")

    train_img = "datasets/detection_dataset/train/images"
    train_lbl = "datasets/detection_dataset/train/labels"

    print("Loading dataset...")

    dataset = IrisDetectionDataset(train_img, train_lbl)

    print("Total training images:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )

    print("DataLoader ready")

    model = IrisDetectorCNN()

    print("Model initialized")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20

    best_loss = float("inf")

    print("Training started")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 0:

                percent = (batch_idx / len(loader)) * 100

                print(f"Batch {batch_idx}/{len(loader)} ({percent:.2f}% done)")

        print("Epoch loss:", running_loss)

        # Save best model
        if running_loss < best_loss:

            best_loss = running_loss

            os.makedirs("ai_models/trained_models", exist_ok=True)

            torch.save(
                model.state_dict(),
                "ai_models/trained_models/iris_detector.pth"
            )

            print("Best model saved")

    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()