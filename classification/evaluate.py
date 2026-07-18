from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CIFAR10_CLASSES, TrainConfig, get_device
from src.dataset import get_dataloaders
from src.engine import collect_predictions
from src.model import build_model

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def plot_training_curves(history: list[dict], path: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, [h["train_loss"] for h in history], "-o", ms=3, label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], "-s", ms=3, label="validation")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, [h["train_acc"] for h in history], "-o", ms=3, label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], "-s", ms=3, label="validation")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    classes = ["non-vehicle", "vehicle"]
    ax.set_xticks([0, 1], labels=classes)
    ax.set_yticks([0, 1], labels=classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(y_true, y_prob, path: Path) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"CNN (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(roc_auc)


def plot_pr(y_true, y_prob, path: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(cfg: TrainConfig, model, device, path: Path, n: int = 12) -> None:
    """Visualize CNN predictions on simple, individual test images."""
    from torchvision import datasets, transforms

    raw = datasets.CIFAR10(root=str(cfg.data_dir), train=False, download=False)
    norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)]
    )
    from src.config import VEHICLE_CLASS_INDICES

    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(raw), size=n, replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    model.eval()
    for ax, idx in zip(axes.ravel(), indices):
        img, label = raw[idx]
        true_bin = 1 if label in VEHICLE_CLASS_INDICES else 0
        with torch.no_grad():
            tensor = norm(img).unsqueeze(0).to(device)
            prob = torch.softmax(model(tensor), dim=1)[0, 1].item()
        pred_bin = 1 if prob >= 0.5 else 0
        ax.imshow(img)
        ax.axis("off")
        true_lbl = "vehicle" if true_bin else "non-veh"
        pred_lbl = "vehicle" if pred_bin else "non-veh"
        color = "green" if pred_bin == true_bin else "red"
        ax.set_title(
            f"{CIFAR10_CLASSES[label]}\ntrue={true_lbl} / pred={pred_lbl}\nP(veh)={prob:.2f}",
            color=color, fontsize=8,
        )
    fig.suptitle("CNN predictions on individual test images", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig()
    device = get_device()
    print(f"Device: {device}")

    history_path = cfg.checkpoint_dir / "history.json"
    with open(history_path) as f:
        hist_data = json.load(f)
    history = hist_data["history"]

    ckpt_path = cfg.checkpoint_dir / "best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, _, test_loader = get_dataloaders(cfg)
    print("Collecting CNN predictions on the test set...")
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device, desc="test")

    cnn_metrics = {
        "model": "VehicleCNN (ours)",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "n_params": int(sum(p.numel() for p in model.parameters())),
    }
    print("CNN test metrics:")
    for k, v in cnn_metrics.items():
        print(f"  {k}: {v}")

    print("Generating figures...")
    plot_training_curves(history, FIG_DIR / "training_curves.png")
    plot_confusion_matrix(y_true, y_pred, FIG_DIR / "confusion_matrix.png")
    plot_roc(y_true, y_prob, FIG_DIR / "roc_curve.png")
    plot_pr(y_true, y_prob, FIG_DIR / "pr_curve.png")
    plot_sample_predictions(cfg, model, device, FIG_DIR / "sample_predictions.png")

    with open(REPORTS_DIR / "evaluation.json", "w") as f:
        json.dump(cnn_metrics, f, indent=2)

    comparison = [cnn_metrics]
    baseline_path = cfg.checkpoint_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            comparison.extend(json.load(f))
    with open(REPORTS_DIR / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nDone. Figures in {FIG_DIR}, tables in {REPORTS_DIR}")


if __name__ == "__main__":
    main()
