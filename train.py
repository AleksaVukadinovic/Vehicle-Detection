"""Entry point for training the vehicle detection CNN.

Example:
    python train.py --epochs 20 --batch-size 128 --lr 1e-3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from src.config import TrainConfig
from src.dataset import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vehicle detection CNN")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print("Loading data (CIFAR-10 will be downloaded on first run)...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    print(
        f"Batches -> train: {len(train_loader)}, "
        f"val: {len(val_loader)}, test: {len(test_loader)}"
    )

    model = build_model(num_classes=2).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs
    )

    best_val_acc = 0.0
    history = []
    best_path = cfg.checkpoint_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, epoch
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, cfg.device, desc="val"
        )
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "config": {
                        "mean": cfg.mean,
                        "std": cfg.std,
                    },
                },
                best_path,
            )
            print(f"  -> Saved new best model (val_acc={val_acc:.4f})")

    # Final test evaluation using the best checkpoint.
    print("\nLoading best model for final test evaluation...")
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, cfg.device, desc="test"
    )
    print(f"Test loss {test_loss:.4f} | Test accuracy {test_acc:.4f}")

    with open(cfg.checkpoint_dir / "history.json", "w") as f:
        json.dump(
            {"history": history, "test_acc": test_acc, "test_loss": test_loss},
            f,
            indent=2,
        )
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Artifacts saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
