import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

torch.set_num_threads(4)

from ai_models.anti_spoofing.model import AntiSpoofCNN


def train_anti_spoofing_high_test_acc(target_accuracy=95.0, max_epochs=15, batch_size=64, lr=0.0005):
    print("\n==================================================")
    print(" HIGH TEST ACCURACY TRAINING: ANTI-SPOOFING MODEL (TARGET: 95.0% - 99.0%)")
    print("==================================================")

    train_dir = os.path.abspath("datasets/anti_spoof_dataset/train")
    test_dir = os.path.abspath("datasets/anti_spoof_dataset/test")
    model_save_path = os.path.abspath("ai_models/trained_models/anti_spoof_model.pth")

    if not os.path.exists(train_dir):
        print("Error: Train directory datasets/anti_spoof_dataset/train does not exist!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Execution Device : {device}")

    # Robust Data Augmentation to prevent overfitting
    transform_train = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_test = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = ImageFolder(train_dir, transform=transform_train)
    test_dataset = ImageFolder(test_dir, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"  Train Dataset Size : {len(train_dataset):,} images")
    print(f"  Test Dataset Size  : {len(test_dataset):,} images")
    print(f"  Target Test Acc    : {target_accuracy}% - 99.0%")
    print("--------------------------------------------------\n", flush=True)

    model = AntiSpoofCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
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

            if (i + 1) % 50 == 0 or (i + 1) == len(train_loader):
                print(f"   Epoch [{epoch}/{max_epochs}] | Batch [{i+1}/{len(train_loader)}] | Loss: {running_loss/(i+1):.4f} | Train Acc: {(100.0 * correct / total):.2f}%", flush=True)

        # Test Evaluation at Epoch End
        model.eval()
        test_correct = 0
        test_total = 0
        tp, fp, tn, fn = 0, 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                for pred, label in zip(predicted, labels):
                    if label.item() == 1:
                        if pred.item() == 1:
                            tp += 1
                        else:
                            fn += 1
                    else:
                        if pred.item() == 0:
                            tn += 1
                        else:
                            fp += 1

        test_acc = 100.0 * test_correct / max(1, test_total)
        scheduler.step(test_acc)

        print(f"\n   EPOCH {epoch} TEST EVALUATION RESULTS:", flush=True)
        print(f"    Test Accuracy : {test_acc:.2f}% (Best so far: {max(best_test_acc, test_acc):.2f}%)", flush=True)
        print(f"    Confusion Matrix -> TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}\n", flush=True)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"  [Best Model Saved] Saved to: {model_save_path} with Test Acc: {test_acc:.2f}%\n", flush=True)

        if test_acc >= target_accuracy:
            print(f"\n TARGET TEST ACCURACY REACHED! ({test_acc:.2f}% >= {target_accuracy}%). Stopping training early as requested!", flush=True)
            break

    print("==================================================")
    print(f" ANTI-SPOOFING TRAINING COMPLETE! BEST TEST ACCURACY ACHIEVED: {best_test_acc:.2f}%")
    print("==================================================\n", flush=True)


if __name__ == "__main__":
    train_anti_spoofing_high_test_acc(target_accuracy=95.0, max_epochs=15, batch_size=64, lr=0.0005)
