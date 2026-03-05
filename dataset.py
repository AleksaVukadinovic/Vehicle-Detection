"""
PASCAL VOC 2007 Vehicle Detection Dataset
==========================================
Downloads the PASCAL VOC 2007 dataset (trainval split) on first run,
caches it locally, and builds a tf.data pipeline for training a
grid-based vehicle detector.

The dataset is filtered to keep only images that contain at least one
vehicle-class object.  Annotations are converted into a dense grid
target tensor suitable for a YOLO-style detector.

Vehicle classes (7):
    aeroplane, bicycle, boat, bus, car, motorbike, train

Grid encoding (per cell of the S×S grid):
    [x, y, w, h, confidence] × B  +  [class probabilities] × C
where (x, y) are offsets relative to the cell, (w, h) are relative to
the full image, and confidence is 1 if an object centre falls in that
cell, 0 otherwise.
"""

import os
import tarfile
import xml.etree.ElementTree as ET

import numpy as np
import requests
import tensorflow as tf

# ── Configuration ────────────────────────────────────────────────────
VOC_URL = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
VOC_TEST_URL = "https://thor.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

IMG_SIZE = 448
GRID_S = 7
NUM_BOXES = 2

VEHICLE_CLASSES = ["aeroplane", "bicycle", "boat", "bus",
                   "car", "motorbike", "train"]
NUM_CLASSES = len(VEHICLE_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(VEHICLE_CLASSES)}

# Per-cell target length: B * 5 + C
CELL_DIM = NUM_BOXES * 5 + NUM_CLASSES


def _download_and_extract(url, dest_dir):
    """Download a tar file and extract it into *dest_dir*."""
    os.makedirs(dest_dir, exist_ok=True)
    tar_path = os.path.join(dest_dir, os.path.basename(url))

    if not os.path.exists(tar_path):
        print(f"  Downloading {url} …")
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(tar_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = min(100, downloaded * 100 // total)
                    mb_done = downloaded / 1e6
                    mb_total = total / 1e6
                    print(f"\r  {mb_done:.1f}/{mb_total:.1f} MB ({pct}%)",
                          end="", flush=True)
        print()

    marker = tar_path + ".extracted"
    if not os.path.exists(marker):
        print(f"  Extracting {tar_path} …")
        with tarfile.open(tar_path) as tf_:
            tf_.extractall(dest_dir)
        open(marker, "w").close()


def download_voc2007(data_dir=DATA_DIR):
    """
    Ensure PASCAL VOC 2007 trainval + test data is available locally.

    Returns the path to the VOCdevkit/VOC2007 directory.
    """
    voc_root = os.path.join(data_dir, "VOCdevkit", "VOC2007")
    if os.path.isdir(voc_root):
        print("✓ VOC 2007 data already present.")
        return voc_root

    print("▶ Downloading PASCAL VOC 2007 …")
    _download_and_extract(VOC_URL, data_dir)
    _download_and_extract(VOC_TEST_URL, data_dir)
    print("✓ VOC 2007 ready.")
    return voc_root


# ── Annotation parsing ───────────────────────────────────────────────

def _parse_voc_annotation(xml_path):
    """
    Parse a PASCAL VOC XML annotation file.

    Returns
    -------
    img_w, img_h : int
    objects : list of (class_name, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.iter("object"):
        cls_name = obj.find("name").text
        difficult = int(obj.find("difficult").text)
        if difficult:
            continue
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append((cls_name, xmin, ymin, xmax, ymax))
    return img_w, img_h, objects


def load_voc_vehicle_data(voc_root, split="trainval"):
    """
    Load image paths and vehicle annotations from VOC 2007.

    Only images containing at least one vehicle object are kept.
    Bounding boxes are normalised to [0, 1] relative to image size.

    Returns
    -------
    image_paths : list[str]
    all_boxes   : list[np.ndarray]  — each (N_obj, 5): [cls_id, cx, cy, w, h]
    """
    imgset_path = os.path.join(voc_root, "ImageSets", "Main", f"{split}.txt")
    img_dir = os.path.join(voc_root, "JPEGImages")
    ann_dir = os.path.join(voc_root, "Annotations")

    with open(imgset_path) as f:
        ids = [line.strip() for line in f if line.strip()]

    image_paths = []
    all_boxes = []

    for img_id in ids:
        xml_path = os.path.join(ann_dir, f"{img_id}.xml")
        img_w, img_h, objects = _parse_voc_annotation(xml_path)

        vehicle_objs = [(cls, x1, y1, x2, y2)
                        for cls, x1, y1, x2, y2 in objects
                        if cls in CLASS_TO_IDX]
        if not vehicle_objs:
            continue

        boxes = []
        for cls_name, x1, y1, x2, y2 in vehicle_objs:
            cls_id = CLASS_TO_IDX[cls_name]
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            w = np.clip(w, 0, 1)
            h = np.clip(h, 0, 1)
            boxes.append([cls_id, cx, cy, w, h])

        image_paths.append(os.path.join(img_dir, f"{img_id}.jpg"))
        all_boxes.append(np.array(boxes, dtype=np.float32))

    print(f"  Loaded {len(image_paths)} images with vehicles from '{split}' split.")
    return image_paths, all_boxes


# ── Grid target encoding ─────────────────────────────────────────────

def encode_target(boxes, S=GRID_S, B=NUM_BOXES, C=NUM_CLASSES):
    """
    Convert a list of object boxes into a (S, S, B*5 + C) target tensor.

    For each object, the grid cell containing its centre is responsible.
    Within that cell:
      - (x, y) are offsets from the cell's top-left corner, in [0, 1]
        relative to cell size.
      - (w, h) are relative to the full image, in [0, 1].
      - confidence = 1 (object present).
      - The class one-hot vector is set.

    If multiple objects fall into the same cell, only the last one is
    kept (a known simplification of the grid approach).

    Parameters
    ----------
    boxes : np.ndarray (N, 5) — [cls_id, cx, cy, w, h]

    Returns
    -------
    target : np.ndarray (S, S, B*5 + C)
    """
    target = np.zeros((S, S, B * 5 + C), dtype=np.float32)

    for obj in boxes:
        cls_id = int(obj[0])
        cx, cy, w, h = obj[1], obj[2], obj[3], obj[4]

        # Which grid cell?
        gi = int(cx * S)
        gj = int(cy * S)
        gi = min(gi, S - 1)
        gj = min(gj, S - 1)

        # Offset within the cell
        x_cell = cx * S - gi
        y_cell = cy * S - gj

        # Fill all B box slots with the same target
        for b in range(B):
            offset = b * 5
            target[gj, gi, offset + 0] = x_cell
            target[gj, gi, offset + 1] = y_cell
            target[gj, gi, offset + 2] = w
            target[gj, gi, offset + 3] = h
            target[gj, gi, offset + 4] = 1.0  # confidence

        # Class one-hot
        target[gj, gi, B * 5 + cls_id] = 1.0

    return target


# ── tf.data pipeline ─────────────────────────────────────────────────

MAX_OBJECTS = 50  # pad / truncate to fixed count for batching


def _load_and_preprocess(img_path, boxes_flat):
    """TF-level: read image, resize, normalise."""
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, boxes_flat


def _encode_target_wrapper(img, boxes_flat):
    """Wraps the NumPy encode_target into a tf.py_function."""
    target = tf.py_function(
        func=lambda bf: encode_target(
            bf.numpy().reshape(-1, 5)[
                :int(np.sum(bf.numpy().reshape(-1, 5)[:, 3] > 0))]),
        inp=[boxes_flat],
        Tout=tf.float32)
    target.set_shape([GRID_S, GRID_S, CELL_DIM])
    return img, target


def build_dataset(image_paths, all_boxes, batch_size=16, shuffle=True):
    """
    Build a tf.data.Dataset yielding (image, target_grid) pairs.

    Parameters
    ----------
    image_paths : list[str]
    all_boxes   : list[np.ndarray], each (N_obj, 5)
    batch_size  : int
    shuffle     : bool

    Returns
    -------
    tf.data.Dataset
    """
    # Pre-encode targets (much faster than py_function per sample)
    targets = np.array([encode_target(b) for b in all_boxes], dtype=np.float32)
    paths = np.array(image_paths)

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), reshuffle_each_iteration=True)

    def _load(path, target):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = img / 255.0
        return img, target

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_train_val_datasets(batch_size=16, data_dir=DATA_DIR):
    """
    Download VOC 2007 (if needed), parse annotations, and return
    train and validation tf.data.Datasets.

    The trainval set is split 90/10 for training and validation.
    """
    voc_root = download_voc2007(data_dir)
    image_paths, all_boxes = load_voc_vehicle_data(voc_root, split="trainval")

    n = len(image_paths)
    indices = np.random.RandomState(42).permutation(n)
    split = int(n * 0.9)

    train_paths = [image_paths[i] for i in indices[:split]]
    train_boxes = [all_boxes[i] for i in indices[:split]]
    val_paths = [image_paths[i] for i in indices[split:]]
    val_boxes = [all_boxes[i] for i in indices[split:]]

    train_ds = build_dataset(train_paths, train_boxes,
                             batch_size=batch_size, shuffle=True)
    val_ds = build_dataset(val_paths, val_boxes,
                           batch_size=batch_size, shuffle=False)

    print(f"  Train: {len(train_paths)} images, Val: {len(val_paths)} images")
    return train_ds, val_ds
