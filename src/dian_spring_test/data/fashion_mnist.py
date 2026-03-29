from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_fashion_mnist_loaders(
    data_root: str | Path,
    batch_size: int = 256,
    val_ratio: float = 0.1,
    num_workers: int = 2,
):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train_dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transform,
    )

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
