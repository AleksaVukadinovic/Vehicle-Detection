from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from torchvision.ops import nms
from src.anchors import box_iou, decode_boxes, generate_anchors
from src.config import DetectionConfig
from src.dataset import get_dataloaders
from src.model import build_model

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FIG_DIR = REPORTS_DIR / "figures"


@torch.no_grad()
def predict_image(model, cfg, anchors, tensor, threshold=0.05):
    obj_logits, reg_preds = model(tensor.unsqueeze(0).to(cfg.device))
    scores = torch.sigmoid(obj_logits[0])
    boxes = decode_boxes(reg_preds[0], anchors.to(cfg.device))
    keep = scores >= threshold
    scores = scores[keep].cpu()
    boxes = boxes[keep].cpu()
    if boxes.numel() == 0:
        return boxes, scores
    keep_idx = nms(boxes, scores, cfg.nms_iou)
    return boxes[keep_idx], scores[keep_idx]


def collect_detections(model, cfg, anchors, val_ds):
    records = []
    total_gt = 0
    for i in range(len(val_ds)):
        tensor, gt_boxes = val_ds[i]
        pred_boxes, pred_scores = predict_image(model, cfg, anchors, tensor)
        total_gt += gt_boxes.size(0)
        matched = torch.zeros(gt_boxes.size(0), dtype=torch.bool)
        for box, score in zip(pred_boxes, pred_scores):
            tp = False
            if gt_boxes.numel() > 0:
                ious = box_iou(box.unsqueeze(0), gt_boxes)[0]
                ious[matched] = 0.0
                best_iou, best_idx = ious.max(dim=0)
                if best_iou >= 0.5:
                    matched[best_idx] = True
                    tp = True
            records.append((float(score), tp))
    return records, total_gt


def compute_ap(records, total_gt):
    records.sort(key=lambda r: r[0], reverse=True)
    tps = np.array([r[1] for r in records], dtype=np.float64)
    scores = np.array([r[0] for r in records])
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(1.0 - tps)
    recall = cum_tp / max(total_gt, 1)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-9)

    mrec = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    mpre = np.concatenate([[1.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap, recall, precision, scores


def metrics_at_threshold(records, total_gt, threshold):
    tp = sum(1 for s, hit in records if s >= threshold and hit)
    fp = sum(1 for s, hit in records if s >= threshold and not hit)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp}


def plot_training_curves(history_path: Path, out_path: Path):
    with open(history_path) as f:
        history = json.load(f)["history"]
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [h["train"]["loss"] for h in history], marker=".", label="train")
    axes[0].plot(epochs, [h["val"]["loss"] for h in history], marker=".", label="validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(epochs, [h["train"]["cls_loss"] for h in history], marker=".", label="classification (train)")
    axes[1].plot(epochs, [h["train"]["reg_loss"] for h in history], marker=".", label="regression (train)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss component")
    axes[1].set_title("Loss components")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(recall, precision, ap, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(recall, precision, color="tab:blue", label=f"detector (AP50 = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (IoU >= 0.5)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_detections(model, cfg, anchors, val_ds, out_path: Path, num_images=6):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, i in zip(axes.flat, range(num_images)):
        image, target = val_ds.base[val_ds.indices[i]]
        tensor, gt_boxes = val_ds[i]
        pred_boxes, pred_scores = predict_image(
            model, cfg, anchors, tensor, threshold=cfg.score_threshold
        )
        w, h = image.size
        ax.imshow(image)
        for box in gt_boxes:
            x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).tolist()
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="lime", facecolor="none",
                )
            )
        for box, score in zip(pred_boxes, pred_scores):
            x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).tolist()
            ax.add_patch(
                patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor="red", facecolor="none",
                )
            )
            ax.text(
                x1, max(y1 - 4, 0), f"{score:.0%}",
                color="white", fontsize=8,
                bbox=dict(facecolor="red", pad=1, edgecolor="none"),
            )
        ax.set_axis_off()
    fig.suptitle("Detections on validation images (green = ground truth, red = prediction)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = DetectionConfig()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = cfg.checkpoint_dir / "best_detector.pt"
    ckpt = torch.load(ckpt_path, map_location=cfg.device)
    model = build_model(cfg).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    anchors = generate_anchors(cfg)

    _, val_loader = get_dataloaders(cfg)
    val_ds = val_loader.dataset
    print(f"Evaluating on {len(val_ds)} validation images...")

    records, total_gt = collect_detections(model, cfg, anchors, val_ds)
    ap, recall, precision, _ = compute_ap(list(records), total_gt)
    at_thr = metrics_at_threshold(records, total_gt, cfg.score_threshold)

    print(f"AP@0.5: {ap:.4f}")
    print(
        f"At threshold {cfg.score_threshold}: "
        f"precision {at_thr['precision']:.4f}, recall {at_thr['recall']:.4f}, "
        f"F1 {at_thr['f1']:.4f} (TP {at_thr['tp']}, FP {at_thr['fp']}, GT {total_gt})"
    )

    plot_training_curves(cfg.checkpoint_dir / "history.json", FIG_DIR / "training_curves.png")
    plot_pr_curve(recall, precision, ap, FIG_DIR / "pr_curve.png")
    plot_sample_detections(model, cfg, anchors, val_ds, FIG_DIR / "sample_detections.png")

    results = {
        "ap50": ap,
        "score_threshold": cfg.score_threshold,
        "at_threshold": at_thr,
        "total_gt": total_gt,
        "num_val_images": len(val_ds),
        "best_epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("val_loss"),
    }
    with open(REPORTS_DIR / "evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Figures saved to {FIG_DIR}, metrics to {REPORTS_DIR / 'evaluation.json'}")


if __name__ == "__main__":
    main()
