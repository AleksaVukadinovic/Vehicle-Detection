# Vehicle-Detection

Project done as part of "Computational Intelligence" course at 4th year at Faculty of Mathematics, University of Belgrade.

## Overview

A **multi-object Vehicle Detection System** implemented **from scratch** using TensorFlow/Keras — no pre-trained models or high-level detection libraries (no YOLO/SSD/ResNet imports).

The system uses a YOLO-v1-inspired grid-based architecture:
- **Custom CNN backbone** — 6 convolutional blocks trained from scratch
- **Grid-based detection** — divides the image into a 7×7 grid; each cell predicts 2 bounding boxes + class probabilities
- **Non-Maximum Suppression** — filters overlapping detections at inference

Trained on **PASCAL VOC 2007** (automatically downloaded on first run), detecting 7 vehicle classes:
`aeroplane`, `bicycle`, `boat`, `bus`, `car`, `motorbike`, `train`

## Project Structure

| File | Description |
|---|---|
| `dataset.py` | PASCAL VOC 2007 download, parsing, grid target encoding, `tf.data` pipeline |
| `model.py` | Custom CNN backbone with grid-based detection head |
| `train.py` | Custom training loop with `tf.GradientTape` and YOLO-style multi-component loss |
| `utils.py` | Grid decoding, NMS, bounding box drawing, image annotation |
| `inference.py` | CLI tool — takes an image, outputs it with all vehicles detected |
| `run.sh` | Convenience script for training and inference |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

```bash
./run.sh train --epochs 50 --batch_size 16 --lr 0.0001
```

Or directly:
```bash
python train.py --epochs 50 --batch_size 16 --lr 0.0001
```

On first run, the PASCAL VOC 2007 dataset (~450 MB) is automatically downloaded and cached in `data/`.

### Training hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 16 | Mini-batch size |
| `--lr` | 0.0001 | Adam learning rate (with cosine decay) |
| `--save_dir` | `outputs/` | Directory for weights and plots |

## Inference

Detect vehicles in any image:

```bash
./run.sh detect path/to/image.jpg --output result.png
```

Or directly:
```bash
python inference.py path/to/image.jpg --output result.png --conf 0.3 --iou 0.4
```

This outputs the input image with bounding boxes and labels drawn on all detected vehicles.

## Method

### Architecture

The model is a single-shot grid-based detector (inspired by YOLO v1):

```
Input (448×448×3)
    → CNN Backbone (6 Conv blocks: Conv→BN→LeakyReLU→MaxPool)
    → 7×7 spatial feature map
    → 1×1 Conv detection head
    → Output: (7, 7, 17)  [2 boxes × 5 values + 7 classes]
```

Each of the 49 grid cells predicts:
- **2 bounding boxes**, each with 5 values: `(x, y, w, h, confidence)`
- **7 class probabilities** (one per vehicle type)

### Loss Function

The YOLO-style multi-component loss:

$$L = \lambda_{coord} \cdot L_{xy} + \lambda_{coord} \cdot L_{wh} + L_{conf}^{obj} + \lambda_{noobj} \cdot L_{conf}^{noobj} + L_{class}$$

| Component | Purpose | Weight |
|---|---|---|
| $L_{xy}$ | Centre coordinate error | λ_coord = 5.0 |
| $L_{wh}$ | Size error (sqrt scale for small-object sensitivity) | λ_coord = 5.0 |
| $L_{conf}^{obj}$ | Confidence for cells with objects | 1.0 |
| $L_{conf}^{noobj}$ | Confidence for empty cells (down-weighted) | λ_noobj = 0.5 |
| $L_{class}$ | Classification error | 1.0 |

### Key Design Decisions

- **sqrt(w), sqrt(h)** in the size loss — penalises errors in small objects more heavily
- **λ_noobj = 0.5** — prevents the vast majority of empty cells from dominating training
- **LeakyReLU (α=0.1)** — avoids dying neurons in cells with no objects
- **Responsible box selection** — only the predicted box with highest IoU against ground truth receives coordinate gradients
- **NMS at inference** — greedily removes overlapping detections per class
