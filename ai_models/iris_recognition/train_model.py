import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim

from ai_models.iris_recognition.model import IrisRecognitionCNN
from ai_models.iris_recognition.dataset_loader import get_dataloaders


print("Loading dataset...")

train_loader, test_loader, num_classes = get_dataloaders()

print("Train classes:", num_classes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


print("\nCreating model...")

model = IrisRecognitionCNN(num_classes)
model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0001   # reduced for better learning
)


epochs = 100   # fresh training
print("EPOCHS SET TO:", epochs)

print("\nTraining started...\n")


for epoch in range(epochs):

    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad() 

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    # ✅ validation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs}  "
        f"Loss: {running_loss:.4f}  "
        f"Test Accuracy: {accuracy:.2f}%"
    )


torch.save(
    model.state_dict(),
    "ai_models/trained_models/iris_recognition_model.pth"
)

print("\nModel saved successfully")