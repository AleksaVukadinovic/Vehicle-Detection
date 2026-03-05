"""
Inference CLI — Detect vehicles in any image
=============================================
Usage:
    python inference.py path/to/image.jpg
    python inference.py path/to/image.jpg --output result.png
    python inference.py path/to/image.jpg --conf 0.25 --iou 0.4
"""

import argparse
import os

import numpy as np
import tensorflow as tf

from dataset import NUM_CLASSES
from model import build_vehicle_detector
from utils import annotate_image, predict, VEHICLE_CLASSES


def main():
    parser = argparse.ArgumentParser(
        description="Detect vehicles in an image and save the annotated result.")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument("--weights", default="outputs/vehicle_detector.weights.h5",
                        help="Path to trained model weights")
    parser.add_argument("--output", default=None,
                        help="Path to save the annotated image (default: output_<input>)")
    parser.add_argument("--conf", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.4,
                        help="NMS IoU threshold (default: 0.4)")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"Error: Image not found: {args.image}")
        return

    if not os.path.isfile(args.weights):
        print(f"Error: Weights not found: {args.weights}")
        print("Run 'python train.py' first to train the model.")
        return

    print("▶ Loading model …")
    model = build_vehicle_detector()
    model.load_weights(args.weights)

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.image))[0]
        args.output = f"output_{base}.png"

    print(f"▶ Running inference on {args.image} …")
    detections = annotate_image(
        model, args.image, save_path=args.output,
        conf_threshold=args.conf, iou_threshold=args.iou)

    if len(detections) == 0:
        print("  No vehicles detected.")
    else:
        print(f"  Found {len(detections)} vehicle(s):")
        for det in detections:
            cls_name = VEHICLE_CLASSES[int(det[5])]
            print(f"    {cls_name:12s}  conf={det[4]:.2f}  "
                  f"box=[{det[0]:.3f}, {det[1]:.3f}, {det[2]:.3f}, {det[3]:.3f}]")


if __name__ == "__main__":
    main()
