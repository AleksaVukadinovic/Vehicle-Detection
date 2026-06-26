"""Training and evaluation loops."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

_DISABLE_TQDM = os.environ.get("DISABLE_TQDM", "0") == "1"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, disable=_DISABLE_TQDM)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=correct / total)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "eval",
) -> tuple[float, float]:
    """Evaluate the model. Returns (avg_loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(loader, desc=f"[{desc}]", leave=False, disable=_DISABLE_TQDM):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "predict",
):
    """Run the model over a loader and collect raw outputs.

    Returns:
        (y_true, y_pred, y_prob) as numpy arrays, where y_prob is the
        probability assigned to the positive (vehicle) class.
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []
    for images, targets in tqdm(loader, desc=f"[{desc}]", leave=False, disable=_DISABLE_TQDM):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)
        all_true.append(targets.cpu().numpy())
        all_pred.append(preds.cpu().numpy())
        all_prob.append(probs.cpu().numpy())
    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )
