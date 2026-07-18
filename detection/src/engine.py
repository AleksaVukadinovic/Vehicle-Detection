from __future__ import annotations
import os
import torch
import torch.nn as nn
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
) -> dict:
    model.train()
    totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0}
    count = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False, disable=_DISABLE_TQDM)
    for images, gt_boxes in pbar:
        images = images.to(device)

        optimizer.zero_grad()
        obj_logits, reg_preds = model(images)
        loss, cls_loss, reg_loss = criterion(obj_logits, reg_preds, gt_boxes)
        loss.backward()
        optimizer.step()

        count += 1
        totals["loss"] += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["reg_loss"] += reg_loss.item()
        pbar.set_postfix(loss=totals["loss"] / count)

    return {k: v / count for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    totals = {"loss": 0.0, "cls_loss": 0.0, "reg_loss": 0.0}
    count = 0

    for images, gt_boxes in tqdm(loader, desc="[val]", leave=False, disable=_DISABLE_TQDM):
        images = images.to(device)
        obj_logits, reg_preds = model(images)
        loss, cls_loss, reg_loss = criterion(obj_logits, reg_preds, gt_boxes)

        count += 1
        totals["loss"] += loss.item()
        totals["cls_loss"] += cls_loss.item()
        totals["reg_loss"] += reg_loss.item()

    return {k: v / count for k, v in totals.items()}
