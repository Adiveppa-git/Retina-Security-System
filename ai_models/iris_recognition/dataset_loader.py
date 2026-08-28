from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloaders():

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),

        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.9,1.1)),

        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    train_dataset = datasets.ImageFolder(
        "datasets/recognition_dataset/train",
        transform=train_transform
    )

    test_dataset = datasets.ImageFolder(
        "datasets/recognition_dataset/test",
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False
    )

    return train_loader, test_loader, len(train_dataset.classes)