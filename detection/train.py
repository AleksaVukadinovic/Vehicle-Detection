from __future__ import annotations
import argparse
import json
import time
import torch
from src.anchors import generate_anchors
from src.config import DetectionConfig
from src.dataset import get_dataloaders
from src.engine import evaluate, train_one_epoch
from src.loss import DetectionLoss
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train single-stage vehicle detector")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-images", type=int, default=1500)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = DetectionConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_train_images=args.max_images,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    torch.manual_seed(cfg.seed)

    print(f"Device: {cfg.device}")
    print("Loading data (VOC2007 will be downloaded on first run)...")
    train_loader, val_loader = get_dataloaders(cfg)
    print(
        f"Images -> train: {len(train_loader.dataset)}, "
        f"val: {len(val_loader.dataset)}"
    )

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    anchors = generate_anchors(cfg)
    criterion = DetectionLoss(cfg, anchors).to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_val_loss = float("inf")
    history = []
    best_path = cfg.checkpoint_dir / "best_detector.pt"

    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, epoch
        )
        val_metrics = evaluate(model, val_loader, criterion, cfg.device)
        scheduler.step()
        elapsed = time.time() - start

        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"train_loss {train_metrics['loss']:.4f} "
            f"(cls {train_metrics['cls_loss']:.4f}, reg {train_metrics['reg_loss']:.4f}) | "
            f"val_loss {val_metrics['loss']:.4f} | "
            f"{elapsed:.1f}s"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_loss": val_metrics["loss"],
                    "epoch": epoch,
                    "config": {
                        "image_size": cfg.image_size,
                        "grid_size": cfg.grid_size,
                        "anchor_scales": cfg.anchor_scales,
                        "anchor_ratios": cfg.anchor_ratios,
                        "mean": cfg.mean,
                        "std": cfg.std,
                    },
                },
                best_path,
            )
            print(f"  -> Saved new best model (val_loss={val_metrics['loss']:.4f})")

    with open(cfg.checkpoint_dir / "history.json", "w") as f:
        json.dump({"history": history, "best_val_loss": best_val_loss}, f, indent=2)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Artifacts saved to: {cfg.checkpoint_dir}")


if __name__ == "__main__":
    main()
