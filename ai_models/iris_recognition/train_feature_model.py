import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import IrisRecognitionCNN
from triplet_dataset import TripletDataset


print("Loading Triplet Dataset...")

dataset = TripletDataset("datasets/recognition_dataset/train")

loader = DataLoader(dataset, batch_size=16, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = IrisRecognitionCNN().to(device)

criterion = nn.TripletMarginLoss(margin=1.0)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

epochs = 20


print("\nTraining Feature Extractor...\n")

for epoch in range(epochs):

    running_loss = 0

    for anchor, positive, negative in loader:

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_out = model(anchor)
        positive_out = model(positive)
        negative_out = model(negative)

        loss = criterion(anchor_out, positive_out, negative_out)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss:.4f}")


torch.save(
    model.state_dict(),
    "ai_models/trained_models/iris_feature_model.pth"
)

print("\n✅ Feature model trained and saved!")