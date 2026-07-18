#!/usr/bin/env python3
"""
Usage example:
    python predict.py --image path/to/car.jpg
    python predict.py --image a.jpg b.jpg --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import TrainConfig, get_device
from src.model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vehicle detection inference")
    parser.add_argument(
        "--image", nargs="+", required=True, help="Path(s) to input image(s)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to a trained model checkpoint",
    )
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    cfg = ckpt.get("config", {})
    mean = cfg.get("mean", (0.4914, 0.4822, 0.4465))
    std = cfg.get("std", (0.2470, 0.2435, 0.2616))
    return model, mean, std


def main() -> None:
    args = parse_args()
    device = get_device()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train the model first with train.py"
        )

    model, mean, std = load_model(ckpt_path, device)

    tf = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    for image_path in args.image:
        path = Path(image_path)
        if not path.exists():
            print(f"{image_path}: ERROR - file not found")
            continue

        image = Image.open(path).convert("RGB")
        tensor = tf(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        vehicle_prob = probs[1].item()
        label = "VEHICLE" if vehicle_prob >= 0.5 else "NOT A VEHICLE"
        print(
            f"{image_path}: {label} "
            f"(vehicle probability = {vehicle_prob:.2%})"
        )


if __name__ == "__main__":
    main()
