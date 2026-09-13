from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def training_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.04),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def evaluation_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_loaders(data_dir: Path, image_size: int, batch_size: int, workers: int):
    train_set = datasets.ImageFolder(data_dir / "train", transform=training_transform(image_size))
    val_set = datasets.ImageFolder(data_dir / "val", transform=evaluation_transform(image_size))
    if train_set.classes != val_set.classes:
        raise ValueError("train and val must contain the same class folders")
    loaders = {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers),
        "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers),
    }
    return loaders, train_set.classes
