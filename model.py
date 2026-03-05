"""
Custom YOLO-style CNN for Multi-Object Vehicle Detection
=========================================================
Architecture overview
---------------------
This is a single-shot grid-based detector inspired by YOLO v1, built
entirely from scratch using tf.keras.layers (no pre-trained weights).

    Input (448 × 448 × 3)
        │
        ▼
    ┌───────────────────────┐
    │   CNN Backbone         │   6 Conv blocks with BatchNorm,
    │   (Conv→BN→LeakyReLU  │   LeakyReLU, and MaxPool.
    │    → MaxPool) × 6     │   Progressively: 448→224→112→56→28→14→7
    └──────────┬────────────┘
               │
        ┌──────┴──────┐
        ▼              ▼
    Flatten         (spatial
        │            info is
    Dense(1024)     preserved
        │            via 7×7
    Reshape          grid)
        │
        ▼
    Output: (7 × 7 × (B*5 + C))

Each of the 7×7 grid cells predicts:
  • B=2 bounding boxes, each with 5 values:
      (x, y)  — centre offset relative to the cell  [0, 1]
      (w, h)  — size relative to the full image      [0, 1]
      conf    — P(object) × IoU(pred, truth)
  • C=7 class probabilities (one per vehicle class)

**Why a grid-based approach?**
Unlike the previous single-object design, real images contain multiple
vehicles.  The grid partitions the image spatially so each cell is
responsible for detecting objects whose centre falls within it.  This
allows the network to predict a variable number of objects in a single
forward pass — no region proposals or sliding windows needed.

**Why LeakyReLU?**
LeakyReLU (α=0.1) prevents "dying neurons" that can occur with
standard ReLU, which is especially important in detection networks
where many grid cells contain no object and would otherwise produce
zero gradients.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from dataset import GRID_S, NUM_BOXES, NUM_CLASSES, CELL_DIM, IMG_SIZE


def _conv_block(x, filters, kernel_size=3, pool=True):
    """Conv2D → BatchNorm → LeakyReLU (→ MaxPool2D)."""
    x = layers.Conv2D(filters, kernel_size, padding="same",
                      use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    if pool:
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
    return x


def build_vehicle_detector(S=GRID_S, B=NUM_BOXES, C=NUM_CLASSES):
    """
    Construct the grid-based vehicle detection model.

    Parameters
    ----------
    S : int   — grid size (S × S)
    B : int   — bounding boxes per cell
    C : int   — number of vehicle classes

    Returns
    -------
    tf.keras.Model
        Input:  (batch, 448, 448, 3)
        Output: (batch, S, S, B*5 + C)
    """
    cell_dim = B * 5 + C
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")

    # ── CNN Backbone ─────────────────────────────────────────────────
    # 6 pooling stages: 448 → 224 → 112 → 56 → 28 → 14 → 7
    # This produces a 7×7 spatial feature map that directly maps to
    # the 7×7 prediction grid — each spatial position corresponds
    # to one grid cell's receptive field.
    x = _conv_block(inputs, 32)    # → 224
    x = _conv_block(x, 64)        # → 112
    x = _conv_block(x, 128)       # → 56
    x = _conv_block(x, 128, pool=False)
    x = _conv_block(x, 256)       # → 28
    x = _conv_block(x, 256, pool=False)
    x = _conv_block(x, 512)       # → 14
    x = _conv_block(x, 512, pool=False)
    x = _conv_block(x, 1024)      # → 7

    # ── Detection Head ───────────────────────────────────────────────
    # Two 1×1 convolutions reduce the channel dimension to the
    # per-cell prediction vector while preserving spatial layout.
    # This is more parameter-efficient than flattening + Dense and
    # naturally maintains the grid structure.
    x = layers.Conv2D(512, 1, padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(cell_dim, 1, padding="same",
                      kernel_initializer="he_normal",
                      name="grid_output")(x)

    # Apply activations: sigmoid for box coords/confidence,
    # and softmax is handled in the loss (from_logits).
    # We leave the output as raw logits and apply sigmoid selectively
    # during inference / loss computation for numerical stability.

    model = Model(inputs=inputs, outputs=x, name="VehicleDetector")
    return model


if __name__ == "__main__":
    model = build_vehicle_detector()
    model.summary()
