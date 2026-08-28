from dataset_loader import get_dataloaders

train_loader, test_loader, num_classes = get_dataloaders()

print("\nDataset loader working correctly")