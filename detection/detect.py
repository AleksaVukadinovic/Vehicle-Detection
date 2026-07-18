#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import nms
from src.anchors import decode_boxes, generate_anchors
from src.config import DetectionConfig, get_device
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle detection inference")
    parser.add_argument(
        "--image", nargs="+", required=True, help="Path(s) to input image(s)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_detector.pt",
        help="Path to a trained detector checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory where annotated images are saved",
    )
    parser.add_argument("--score-threshold", type=float, default=None)
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved = ckpt.get("config", {})
    cfg = DetectionConfig(
        image_size=saved.get("image_size", 128),
        grid_size=saved.get("grid_size", 16),
        anchor_scales=tuple(saved.get("anchor_scales", (0.25, 0.5, 0.8))),
        anchor_ratios=tuple(saved.get("anchor_ratios", (0.5, 1.0, 2.0))),
        mean=tuple(saved.get("mean", (0.485, 0.456, 0.406))),
        std=tuple(saved.get("std", (0.229, 0.224, 0.225))),
        device=device,
    )
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def detect(model, cfg: DetectionConfig, anchors: torch.Tensor, image: Image.Image, threshold: float):
    tf = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )
    tensor = tf(image).unsqueeze(0).to(cfg.device)

    obj_logits, reg_preds = model(tensor)
    scores = torch.sigmoid(obj_logits[0])
    boxes = decode_boxes(reg_preds[0], anchors.to(cfg.device))

    keep = scores >= threshold
    scores = scores[keep].cpu()
    boxes = boxes[keep].cpu()
    if boxes.numel() == 0:
        return boxes, scores

    keep_idx = nms(boxes, scores, cfg.nms_iou)
    return boxes[keep_idx], scores[keep_idx]


def draw_detections(image: Image.Image, boxes: torch.Tensor, scores: torch.Tensor) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size
    line_width = max(2, round(min(w, h) / 150))
    try:
        font = ImageFont.truetype("Arial.ttf", size=max(12, min(w, h) // 25))
    except OSError:
        font = ImageFont.load_default()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=line_width)
        label = f"vozilo {score:.0%}"
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_h = text_bbox[3] - text_bbox[1]
        text_y = max(0, y1 - text_h - 2 * line_width)
        draw.rectangle(
            [x1, text_y, x1 + (text_bbox[2] - text_bbox[0]) + 2 * line_width, text_y + text_h + 2 * line_width],
            fill="red",
        )
        draw.text((x1 + line_width, text_y + line_width), label, fill="white", font=font)

    return annotated


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train the model first with train.py"
        )

    model, cfg = load_model(ckpt_path, device)
    anchors = generate_anchors(cfg)
    threshold = args.score_threshold if args.score_threshold is not None else cfg.score_threshold

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in args.image:
        path = Path(image_path)
        if not path.exists():
            print(f"{image_path}: ERROR - file not found")
            continue

        image = Image.open(path).convert("RGB")
        boxes, scores = detect(model, cfg, anchors, image, threshold)

        annotated = draw_detections(image, boxes, scores)
        out_path = output_dir / f"{path.stem}_detected{path.suffix}"
        annotated.save(out_path)

        print(f"{image_path}: {len(scores)} vehicle(s) detected -> {out_path}")
        for box, score in zip(boxes, scores):
            w, h = image.size
            x1, y1, x2, y2 = (box * torch.tensor([w, h, w, h])).round().int().tolist()
            print(f"  box=({x1}, {y1}, {x2}, {y2}) score={score:.2%}")


if __name__ == "__main__":
    main()
