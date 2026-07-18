from __future__ import annotations
import random
import tarfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import VOCDetection
from .config import VEHICLE_CLASSES, DetectionConfig

VOC_ARCHIVES = [
    (
        "VOCtrainval_06-Nov-2007.tar",
        "ImageSets/Main/trainval.txt",
        [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar",
        ],
    ),
    (
        "VOCtest_06-Nov-2007.tar",
        "ImageSets/Main/test.txt",
        [
            "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
            "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
        ],
    ),
]


def ensure_voc(data_dir: Path) -> None:
    voc_root = data_dir / "VOCdevkit" / "VOC2007"
    for tar_name, marker, mirrors in VOC_ARCHIVES:
        if (voc_root / marker).exists():
            continue
        tar_path = data_dir / tar_name
        if not tar_path.exists():
            last_error = None
            for url in mirrors:
                try:
                    print(f"Downloading {tar_name} from {url} ...")
                    urllib.request.urlretrieve(url, tar_path)
                    last_error = None
                    break
                except Exception as e:
                    last_error = e
                    tar_path.unlink(missing_ok=True)
            if last_error is not None:
                raise RuntimeError(f"Could not download {tar_name}: {last_error}")
        print(f"Extracting {tar_name} ...")
        with tarfile.open(tar_path) as tar:
            tar.extractall(data_dir)


def parse_vehicle_boxes(target: dict) -> list[list[float]]:
    objects = target["annotation"]["object"]
    if isinstance(objects, dict):
        objects = [objects]
    boxes = []
    for obj in objects:
        if obj["name"] not in VEHICLE_CLASSES:
            continue
        if obj.get("difficult", "0") == "1":
            continue
        bb = obj["bndbox"]
        boxes.append(
            [float(bb["xmin"]), float(bb["ymin"]), float(bb["xmax"]), float(bb["ymax"])]
        )
    return boxes


class VOCVehicleDataset(Dataset):
    def __init__(self, cfg: DetectionConfig, indices: list[int], base: Dataset, train: bool) -> None:
        self.cfg = cfg
        self.indices = indices
        self.base = base
        self.train = train
        self.normalize = transforms.Normalize(cfg.mean, cfg.std)
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.05)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, target = self.base[self.indices[idx]]
        boxes = torch.tensor(parse_vehicle_boxes(target), dtype=torch.float32)

        if self.train:
            image = self.color_jitter(image)
            if random.random() < 0.5:
                w, h = image.size
                scale = random.uniform(1.1, 1.6)
                nw, nh = int(w * scale), int(h * scale)
                fill = tuple(int(255 * m) for m in self.cfg.mean)
                canvas = Image.new("RGB", (nw, nh), fill)
                ox = random.randint(0, nw - w)
                oy = random.randint(0, nh - h)
                canvas.paste(image, (ox, oy))
                image = canvas
                boxes[:, [0, 2]] += ox
                boxes[:, [1, 3]] += oy

        w, h = image.size
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            boxes = torch.stack(
                [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], dim=1
            )

        image = image.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        tensor = transforms.functional.to_tensor(image)
        tensor = self.normalize(tensor)
        return tensor, boxes


def collate_detection(batch):
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    return images, boxes


def get_dataloaders(cfg: DetectionConfig):
    ensure_voc(cfg.data_dir)
    bases = [
        VOCDetection(root=str(cfg.data_dir), year="2007", image_set=split, download=False)
        for split in ("trainval", "test")
    ]
    combined = ConcatDataset(bases)

    vehicle_indices = []
    offset = 0
    for base in bases:
        for i in range(len(base)):
            target = base.parse_voc_xml(ET.parse(base.annotations[i]).getroot())
            if parse_vehicle_boxes(target):
                vehicle_indices.append(offset + i)
        offset += len(base)

    rng = random.Random(cfg.seed)
    rng.shuffle(vehicle_indices)
    vehicle_indices = vehicle_indices[: cfg.max_train_images]

    val_size = max(1, int(len(vehicle_indices) * cfg.val_split))
    val_indices = vehicle_indices[:val_size]
    train_indices = vehicle_indices[val_size:]

    train_ds = VOCVehicleDataset(cfg, train_indices, combined, train=True)
    val_ds = VOCVehicleDataset(cfg, val_indices, combined, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_detection,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_detection,
    )
    return train_loader, val_loader
