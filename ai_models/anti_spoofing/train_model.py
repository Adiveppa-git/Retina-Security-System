import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
torch.set_num_threads(4)

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from ai_models.anti_spoofing.model import AntiSpoofCNN
from ai_models.anti_spoofing.dataset_loader import IrisSpoofDataset


def train():

    print("Starting Anti-Spoof Training System")

    # TRAIN DATASET
    real_path = "datasets/anti_spoof_dataset/train/real"
    fake_path = "datasets/anti_spoof_dataset/train/fake"

    print("Creating dataset...")

    dataset = IrisSpoofDataset(real_path, fake_path)

    print("Total images:", len(dataset))

    print("Creating DataLoader...")

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print("DataLoader ready")

    model = AntiSpoofCNN()

    print("Model initialized")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 10

    print("Starting training loop")

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs} started")

        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(loader):

            if batch_idx == 0:
                print("First batch loaded successfully")

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 200 == 0:
                percent = (batch_idx / len(loader)) * 100
                print(f"Batch {batch_idx}/{len(loader)} ({percent:.2f}% done)")

        print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")

    print("\nSaving trained model...")

    os.makedirs("ai_models/trained_models", exist_ok=True)

    torch.save(model.state_dict(), "ai_models/trained_models/anti_spoof_model.pth")

    print("Model saved successfully")
    print("Training completed successfully")


if __name__ == "__main__":
    train()