"""
Custom Training Loop for the Grid-Based Vehicle Detector
=========================================================
Implements training with tf.GradientTape using a YOLO-v1-style
multi-component loss function.

Loss Function
-------------
The total loss has five components, following the YOLO v1 formulation:

    L = λ_coord · L_xy           (centre coordinate error)
      + λ_coord · L_wh           (width/height error — sqrt scale)
      + L_conf_obj               (confidence for cells WITH objects)
      + λ_noobj · L_conf_noobj   (confidence for cells WITHOUT objects)
      + L_class                  (classification error)

**Why sqrt(w) and sqrt(h)?**
Small deviations in large boxes matter less than in small boxes.
Taking the square root compresses the range so the loss penalises
errors in small objects more heavily, improving detection of smaller
vehicles like bicycles and motorbikes.

**Why λ_noobj < 1?**
Most grid cells contain no object.  Without down-weighting, the
"no object" confidence loss would dominate training, pushing all
confidence scores toward zero.  λ_noobj = 0.5 balances this.

**Why λ_coord = 5?**
Localisation accuracy is critical for detection.  Increasing the
coordinate loss weight ensures the network prioritises getting
bounding boxes right over background confidence.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import (get_train_val_datasets, GRID_S, NUM_BOXES,
                     NUM_CLASSES, CELL_DIM, IMG_SIZE)
from model import build_vehicle_detector

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5


def compute_iou(box1, box2):
    """
    Compute IoU between two sets of boxes in (cx, cy, w, h) format.
    All coordinates are normalised [0, 1].
    """
    # Convert to corners
    b1_x1 = box1[..., 0] - box1[..., 2] / 2
    b1_y1 = box1[..., 1] - box1[..., 3] / 2
    b1_x2 = box1[..., 0] + box1[..., 2] / 2
    b1_y2 = box1[..., 1] + box1[..., 3] / 2

    b2_x1 = box2[..., 0] - box2[..., 2] / 2
    b2_y1 = box2[..., 1] - box2[..., 3] / 2
    b2_x2 = box2[..., 0] + box2[..., 2] / 2
    b2_y2 = box2[..., 1] + box2[..., 3] / 2

    inter_x1 = tf.maximum(b1_x1, b2_x1)
    inter_y1 = tf.maximum(b1_y1, b2_y1)
    inter_x2 = tf.minimum(b1_x2, b2_x2)
    inter_y2 = tf.minimum(b1_y2, b2_y2)

    inter_area = tf.maximum(inter_x2 - inter_x1, 0) * tf.maximum(inter_y2 - inter_y1, 0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-7)


def yolo_loss(predictions, targets, S=GRID_S, B=NUM_BOXES, C=NUM_CLASSES):
    """
    Compute the YOLO v1 multi-component detection loss.

    Parameters
    ----------
    predictions : (batch, S, S, B*5 + C) — raw model output
    targets     : (batch, S, S, B*5 + C) — encoded ground truth

    Returns
    -------
    total_loss, xy_loss, wh_loss, conf_obj_loss, conf_noobj_loss, class_loss
    """
    # Object mask: 1 where an object centre exists, 0 otherwise
    # Use the confidence of the first box slot in the target
    obj_mask = targets[..., 4]  # (batch, S, S)
    noobj_mask = 1.0 - obj_mask

    # ── Parse predictions ────────────────────────────────────────────
    pred_boxes = []
    for b in range(B):
        offset = b * 5
        xy = tf.sigmoid(predictions[..., offset:offset + 2])
        wh = predictions[..., offset + 2:offset + 4]
        conf = tf.sigmoid(predictions[..., offset + 4])
        pred_boxes.append((xy, wh, conf))

    pred_class_logits = predictions[..., B * 5:]

    # ── Parse targets ────────────────────────────────────────────────
    target_xy = targets[..., 0:2]
    target_wh = targets[..., 2:4]
    target_class = targets[..., B * 5:]

    # ── Responsible box selection ────────────────────────────────────
    # For each cell with an object, pick the predicted box with the
    # highest IoU with the ground truth.  Only that box receives
    # coordinate and confidence-object gradients.

    # Build absolute coords for IoU computation
    # Create grid offsets
    grid_y = tf.cast(tf.reshape(tf.range(S), [1, S, 1]), tf.float32)
    grid_x = tf.cast(tf.reshape(tf.range(S), [1, 1, S]), tf.float32)

    ious = []
    for b in range(B):
        xy, wh, _ = pred_boxes[b]
        pred_cx = (xy[..., 0] + grid_x) / S
        pred_cy = (xy[..., 1] + grid_y) / S
        pred_w = tf.abs(wh[..., 0])
        pred_h = tf.abs(wh[..., 1])
        pred_abs = tf.stack([pred_cx, pred_cy, pred_w, pred_h], axis=-1)

        tgt_cx = (target_xy[..., 0] + grid_x) / S
        tgt_cy = (target_xy[..., 1] + grid_y) / S
        tgt_abs = tf.stack([tgt_cx, tgt_cy, target_wh[..., 0], target_wh[..., 1]], axis=-1)

        ious.append(compute_iou(pred_abs, tgt_abs))

    ious_stacked = tf.stack(ious, axis=-1)  # (batch, S, S, B)
    best_box = tf.argmax(ious_stacked, axis=-1)  # (batch, S, S)
    best_box_mask = tf.one_hot(best_box, B)  # (batch, S, S, B)

    # ── Coordinate loss ──────────────────────────────────────────────
    xy_loss = 0.0
    wh_loss = 0.0
    conf_obj_loss = 0.0
    conf_noobj_loss = 0.0

    for b in range(B):
        xy, wh, conf = pred_boxes[b]
        responsible = best_box_mask[..., b] * obj_mask  # (batch, S, S)

        # XY loss
        xy_err = tf.reduce_sum(tf.square(xy - target_xy), axis=-1)
        xy_loss += tf.reduce_sum(responsible * xy_err)

        # WH loss (sqrt scale)
        pred_w_sqrt = tf.sign(wh[..., 0]) * tf.sqrt(tf.abs(wh[..., 0]) + 1e-7)
        pred_h_sqrt = tf.sign(wh[..., 1]) * tf.sqrt(tf.abs(wh[..., 1]) + 1e-7)
        tgt_w_sqrt = tf.sqrt(target_wh[..., 0] + 1e-7)
        tgt_h_sqrt = tf.sqrt(target_wh[..., 1] + 1e-7)
        wh_err = tf.square(pred_w_sqrt - tgt_w_sqrt) + tf.square(pred_h_sqrt - tgt_h_sqrt)
        wh_loss += tf.reduce_sum(responsible * wh_err)

        # Confidence loss (object cells — responsible box)
        # Target confidence = IoU between pred and ground truth
        target_conf = ious[b]
        conf_obj_loss += tf.reduce_sum(
            responsible * tf.square(conf - target_conf))

        # Confidence loss (no-object cells + non-responsible boxes in obj cells)
        noobj_responsible = (1.0 - responsible) * (1.0 - obj_mask) + \
                            (1.0 - best_box_mask[..., b]) * obj_mask
        conf_noobj_loss += tf.reduce_sum(
            noobj_responsible * tf.square(conf - 0.0))

    batch_size = tf.cast(tf.shape(predictions)[0], tf.float32)

    xy_loss = LAMBDA_COORD * xy_loss / batch_size
    wh_loss = LAMBDA_COORD * wh_loss / batch_size
    conf_obj_loss = conf_obj_loss / batch_size
    conf_noobj_loss = LAMBDA_NOOBJ * conf_noobj_loss / batch_size

    # ── Classification loss ──────────────────────────────────────────
    # Only cells with objects contribute to classification loss.
    obj_mask_expanded = tf.expand_dims(obj_mask, -1)
    class_loss = tf.reduce_sum(
        obj_mask_expanded * tf.square(
            tf.nn.softmax(pred_class_logits) - target_class)
    ) / batch_size

    total_loss = xy_loss + wh_loss + conf_obj_loss + conf_noobj_loss + class_loss

    return total_loss, xy_loss, wh_loss, conf_obj_loss, conf_noobj_loss, class_loss


@tf.function
def train_step(model, optimizer, images, targets):
    """Single gradient-tape training step."""
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        total_loss, xy_loss, wh_loss, conf_obj, conf_noobj, cls_loss = \
            yolo_loss(predictions, targets)

    gradients = tape.gradient(total_loss, model.trainable_variables)
    # Clip gradients to prevent explosion
    gradients = [tf.clip_by_norm(g, 5.0) if g is not None else g
                 for g in gradients]
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, xy_loss, wh_loss, conf_obj, conf_noobj, cls_loss


@tf.function
def val_step(model, images, targets):
    """Single validation step."""
    predictions = model(images, training=False)
    return yolo_loss(predictions, targets)


def train(epochs=50, batch_size=16, learning_rate=1e-4,
          save_dir="outputs"):
    """
    Full training procedure.

    1. Download PASCAL VOC 2007 (if not cached).
    2. Build the grid-based detector from scratch.
    3. Train with custom YOLO loss for *epochs* epochs.
    4. Save weights and training curves.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("▶ Preparing dataset …")
    train_ds, val_ds = get_train_val_datasets(batch_size=batch_size)

    print("▶ Building model …")
    model = build_vehicle_detector()
    model.summary()

    # Learning rate schedule: warm-up then cosine decay
    total_steps = epochs * 100  # approximate
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        alpha=1e-6)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    history = {
        "train_loss": [], "val_loss": [],
        "train_xy": [], "train_wh": [],
        "train_conf_obj": [], "train_conf_noobj": [],
        "train_cls": [],
    }

    print("▶ Training …\n")
    for epoch in range(1, epochs + 1):
        # ── Training ─────────────────────────────────────────────────
        t_total, t_xy, t_wh, t_cobj, t_cno, t_cls, steps = \
            0., 0., 0., 0., 0., 0., 0
        for images, targets in train_ds:
            loss, xy, wh, cobj, cno, cls = train_step(
                model, optimizer, images, targets)
            t_total += loss.numpy()
            t_xy += xy.numpy()
            t_wh += wh.numpy()
            t_cobj += cobj.numpy()
            t_cno += cno.numpy()
            t_cls += cls.numpy()
            steps += 1

        # ── Validation ───────────────────────────────────────────────
        v_total, v_steps = 0., 0
        for images, targets in val_ds:
            loss, *_ = val_step(model, images, targets)
            v_total += loss.numpy()
            v_steps += 1

        # ── Log ──────────────────────────────────────────────────────
        history["train_loss"].append(t_total / steps)
        history["val_loss"].append(v_total / max(v_steps, 1))
        history["train_xy"].append(t_xy / steps)
        history["train_wh"].append(t_wh / steps)
        history["train_conf_obj"].append(t_cobj / steps)
        history["train_conf_noobj"].append(t_cno / steps)
        history["train_cls"].append(t_cls / steps)

        print(f"Epoch {epoch:3d}/{epochs}  │  "
              f"loss {history['train_loss'][-1]:7.3f}  "
              f"xy {history['train_xy'][-1]:.3f}  "
              f"wh {history['train_wh'][-1]:.3f}  "
              f"conf {history['train_conf_obj'][-1]:.3f}/{history['train_conf_noobj'][-1]:.3f}  "
              f"cls {history['train_cls'][-1]:.3f}  │  "
              f"val {history['val_loss'][-1]:7.3f}")

    # ── Save ─────────────────────────────────────────────────────────
    weights_path = os.path.join(save_dir, "vehicle_detector.weights.h5")
    model.save_weights(weights_path)
    print(f"\n✓ Weights saved to {weights_path}")

    _plot_history(history, save_dir)
    print(f"✓ Training curves saved to {save_dir}/")

    return model, history


def _plot_history(history, save_dir):
    """Save training curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_xy"], label="XY")
    axes[1].plot(epochs, history["train_wh"], label="WH")
    axes[1].plot(epochs, history["train_cls"], label="Class")
    axes[1].set_title("Component Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_conf_obj"], label="Conf (obj)")
    axes[2].plot(epochs, history["train_conf_noobj"], label="Conf (noobj)")
    axes[2].set_title("Confidence Losses")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Vehicle Detector")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="outputs")
    args = parser.parse_args()

    train(epochs=args.epochs,
          batch_size=args.batch_size,
          learning_rate=args.lr,
          save_dir=args.save_dir)
