from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from .config import VEHICLE_CLASS_INDICES, TrainConfig

# just a wrapper, converts labels to 1/0 depending if the image is vehicle or no
class VehicleBinaryDataset(Dataset):
    def __init__(self, base: Dataset) -> None:
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        target = 1 if label in VEHICLE_CLASS_INDICES else 0
        return image, target


def build_transforms(cfg: TrainConfig):
    """Return (train_transform, eval_transform)."""
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    return train_tf, eval_tf


def get_dataloaders(cfg: TrainConfig):
    """Create train, validation, and test dataloaders.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf, eval_tf = build_transforms(cfg)

    full_train = datasets.CIFAR10(
        root=str(cfg.data_dir), train=True, download=True, transform=train_tf
    )
    test_raw = datasets.CIFAR10(
        root=str(cfg.data_dir), train=False, download=True, transform=eval_tf
    )

    val_size = int(len(full_train) * cfg.val_split)
    train_size = len(full_train) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_subset, val_subset = random_split(
        full_train, [train_size, val_size], generator=generator
    )

    val_base = datasets.CIFAR10(
        root=str(cfg.data_dir), train=True, download=False, transform=eval_tf
    )
    val_subset.dataset = val_base

    train_ds = VehicleBinaryDataset(train_subset)
    val_ds = VehicleBinaryDataset(val_subset)
    test_ds = VehicleBinaryDataset(test_raw)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
    )
    return train_loader, val_loader, test_loader


def get_numpy_arrays(cfg: TrainConfig):
    train_raw = datasets.CIFAR10(root=str(cfg.data_dir), train=True, download=True)
    test_raw = datasets.CIFAR10(root=str(cfg.data_dir), train=False, download=True)

    def to_xy(ds):
        x = ds.data.astype(np.float32).reshape(len(ds), -1) / 255.0
        y = np.array(
            [1 if lab in VEHICLE_CLASS_INDICES else 0 for lab in ds.targets],
            dtype=np.int64,
        )
        return x, y

    x_train, y_train = to_xy(train_raw)
    x_test, y_test = to_xy(test_raw)
    return x_train, y_train, x_test, y_test
