"""
Inference & Visualisation Utilities
====================================
Provides:
  1. Grid decoding — convert model output tensor to a list of detections.
  2. Non-Maximum Suppression (NMS) — filter overlapping boxes.
  3. Drawing — annotate an image with all detected vehicles.
  4. High-level predict function for single images.
"""

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dataset import (IMG_SIZE, GRID_S, NUM_BOXES, NUM_CLASSES,
                     VEHICLE_CLASSES)

COLORS = {
    "aeroplane": "#e74c3c",
    "bicycle":   "#e67e22",
    "boat":      "#3498db",
    "bus":       "#9b59b6",
    "car":       "#2ecc71",
    "motorbike": "#f1c40f",
    "train":     "#1abc9c",
}


def decode_predictions(output, S=GRID_S, B=NUM_BOXES, C=NUM_CLASSES,
                       conf_threshold=0.3):
    """
    Decode the raw model output (S, S, B*5+C) into a list of detections.

    Each detection is [x_min, y_min, x_max, y_max, confidence, class_id].
    Coordinates are normalised to [0, 1].

    Parameters
    ----------
    output : np.ndarray (S, S, B*5 + C) — single image prediction
    conf_threshold : float — minimum confidence to keep a detection

    Returns
    -------
    detections : np.ndarray (N, 6)
    """
    detections = []

    for j in range(S):
        for i in range(S):
            cell = output[j, i]

            # Class probabilities
            class_logits = cell[B * 5:]
            class_probs = _softmax(class_logits)
            best_class = np.argmax(class_probs)
            best_class_prob = class_probs[best_class]

            for b in range(B):
                offset = b * 5
                x_cell = _sigmoid(cell[offset + 0])
                y_cell = _sigmoid(cell[offset + 1])
                w_raw = cell[offset + 2]
                h_raw = cell[offset + 3]
                conf = _sigmoid(cell[offset + 4])

                # Combined score = P(object) × P(class | object)
                score = conf * best_class_prob
                if score < conf_threshold:
                    continue

                # Convert cell-relative coords to image-relative
                cx = (i + x_cell) / S
                cy = (j + y_cell) / S
                w = abs(w_raw)
                h = abs(h_raw)

                x_min = max(0, cx - w / 2)
                y_min = max(0, cy - h / 2)
                x_max = min(1, cx + w / 2)
                y_max = min(1, cy + h / 2)

                detections.append([x_min, y_min, x_max, y_max,
                                   score, best_class])

    if not detections:
        return np.zeros((0, 6))
    return np.array(detections)


def nms(detections, iou_threshold=0.4):
    """
    Non-Maximum Suppression.

    Greedily selects high-scoring detections and removes overlapping
    boxes of the same class that exceed the IoU threshold.

    Parameters
    ----------
    detections : np.ndarray (N, 6) — [x1, y1, x2, y2, score, class_id]
    iou_threshold : float

    Returns
    -------
    np.ndarray (M, 6) — filtered detections
    """
    if len(detections) == 0:
        return detections

    kept = []
    classes = np.unique(detections[:, 5])

    for cls in classes:
        cls_mask = detections[:, 5] == cls
        cls_dets = detections[cls_mask]

        # Sort by score descending
        order = np.argsort(-cls_dets[:, 4])
        cls_dets = cls_dets[order]

        selected = []
        while len(cls_dets) > 0:
            best = cls_dets[0]
            selected.append(best)
            if len(cls_dets) == 1:
                break

            rest = cls_dets[1:]
            ious = _compute_iou_np(best[:4], rest[:, :4])
            cls_dets = rest[ious < iou_threshold]

        kept.extend(selected)

    return np.array(kept) if kept else np.zeros((0, 6))


def _compute_iou_np(box, boxes):
    """Compute IoU between one box and an array of boxes (corner format)."""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    return inter / (area_a + area_b - inter + 1e-7)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _softmax(x):
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-7)


def predict(model, image, conf_threshold=0.3, iou_threshold=0.4):
    """
    Run full inference on a single image.

    Parameters
    ----------
    model : tf.keras.Model
    image : np.ndarray (H, W, 3), values in [0, 255] or [0, 1]
    conf_threshold : float
    iou_threshold : float

    Returns
    -------
    detections : np.ndarray (N, 6) — [x1, y1, x2, y2, score, class_id]
    """
    if image.max() > 1.0:
        image = image / 255.0

    img = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = tf.expand_dims(img, 0)

    output = model(img, training=False).numpy()[0]
    dets = decode_predictions(output, conf_threshold=conf_threshold)
    dets = nms(dets, iou_threshold=iou_threshold)
    return dets


def draw_detections(image, detections, ax=None):
    """
    Draw all detected vehicles on the image.

    Parameters
    ----------
    image      : np.ndarray (H, W, 3)
    detections : np.ndarray (N, 6) — [x1, y1, x2, y2, score, class_id]
    ax         : matplotlib Axes (created if None)

    Returns
    -------
    ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if image.max() > 1.0:
        image = image / 255.0

    h, w = image.shape[:2]
    ax.imshow(image)

    for det in detections:
        x1, y1, x2, y2, score, cls_id = det
        cls_name = VEHICLE_CLASSES[int(cls_id)]
        color = COLORS.get(cls_name, "#ffffff")

        rect = patches.Rectangle(
            (x1 * w, y1 * h), (x2 - x1) * w, (y2 - y1) * h,
            linewidth=2, edgecolor=color, facecolor="none")
        ax.add_patch(rect)

        ax.text(x1 * w, y1 * h - 4,
                f"{cls_name} {score:.0%}",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor=color, alpha=0.85, pad=2,
                          edgecolor="none"))

    ax.axis("off")
    return ax


def annotate_image(model, image_path, save_path=None,
                   conf_threshold=0.3, iou_threshold=0.4):
    """
    Load an image, detect vehicles, draw boxes, and save/show.

    Parameters
    ----------
    model       : trained tf.keras.Model
    image_path  : str — path to the input image
    save_path   : str — if provided, save the annotated image here
    conf_threshold : float
    iou_threshold  : float

    Returns
    -------
    detections : np.ndarray (N, 6)
    """
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(raw, channels=3).numpy()
    image_float = image.astype(np.float32) / 255.0

    detections = predict(model, image_float,
                         conf_threshold=conf_threshold,
                         iou_threshold=iou_threshold)

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    draw_detections(image_float, detections, ax=ax)

    n = len(detections)
    ax.set_title(f"Detected {n} vehicle{'s' if n != 1 else ''}", fontsize=14)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Annotated image saved to {save_path}")
    else:
        plt.savefig("output.png", dpi=150, bbox_inches="tight")
        print("✓ Annotated image saved to output.png")
    plt.close()

    return detections
