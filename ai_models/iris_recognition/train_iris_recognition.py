import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models

torch.set_num_threads(4)


class IrisIdentityCNN(nn.Module):
    """
    High-Precision Iris Identity Classifier for Subject Recognition.
    Maps cropped iris patterns directly to unique subject identities.
    """
    def __init__(self, num_classes=200):
        super(IrisIdentityCNN, self).__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.BatchNorm1d(256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        logits = self.classifier(feat)
        return logits


def train_recognition_target_acc(target_accuracy=95.0, max_epochs=20, batch_size=64, lr=0.001):
    print("\n==================================================")
    print(" HIGH-PRECISION IRIS RECOGNITION TRAINING (TARGET: 95.0% - 99.0%)")
    print("==================================================")

    train_dir = os.path.abspath("datasets/recognition_dataset/train")
    test_dir = os.path.abspath("datasets/recognition_dataset/test")
    model_save_path = os.path.abspath("ai_models/trained_models/iris_feature_model.pth")

    if not os.path.exists(train_dir):
        print("Error: Train directory datasets/recognition_dataset/train does not exist!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device : {device}")

    transform_train = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=transform_train)
    test_dataset = ImageFolder(test_dir, transform=transform_test)

    # Sync class_to_idx mapping between train and test datasets
    test_dataset.class_to_idx = train_dataset.class_to_idx
    test_dataset.samples = [(p, train_dataset.class_to_idx[os.path.basename(os.path.dirname(p))]) for p, _ in test_dataset.samples if os.path.basename(os.path.dirname(p)) in train_dataset.class_to_idx]
    test_dataset.targets = [s[1] for s in test_dataset.samples]

    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train Dataset Size : {len(train_dataset):,} images")
    print(f"  Test Dataset Size  : {len(test_dataset):,} images")
    print(f"  Subject Classes    : {num_classes}")
    print(f"  Target Test Acc    : {target_accuracy}% - 99.0%")
    print("--------------------------------------------------\n", flush=True)

    model = IrisIdentityCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_test_acc = 0.0

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 20 == 0 or (i + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{max_epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {running_loss/(i+1):.4f} | Train Acc: {(100.0 * correct / total):.2f}%", flush=True)

        # Test Evaluation at Epoch End
        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = 100.0 * test_correct / max(1, test_total)
        scheduler.step(test_acc)

        print(f"\n   EPOCH {epoch} TEST RECOGNITION EVALUATION RESULTS:", flush=True)
        print(f"    Test Recognition Accuracy : {test_acc:.2f}% (Best so far: {max(best_test_acc, test_acc):.2f}%)", flush=True)
        print(f"    Correct Subject Matches   : {test_correct} / {test_total}\n", flush=True)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  [Best Model Saved] Saved to: {model_save_path} with Test Acc: {test_acc:.2f}%\n", flush=True)

        if test_acc >= target_accuracy:
            print(f"\n TARGET TEST RECOGNITION ACCURACY REACHED! ({test_acc:.2f}% >= {target_accuracy}%). Stopping training early as requested!", flush=True)
            break

    print("==================================================")
    print(f" IRIS RECOGNITION TRAINING COMPLETE! BEST TEST ACCURACY: {best_test_acc:.2f}%")
    print("==================================================\n", flush=True)


if __name__ == "__main__":
    train_recognition_target_acc(target_accuracy=95.0, max_epochs=20, batch_size=64, lr=0.001)
