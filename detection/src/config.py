from dataclasses import dataclass, field
from pathlib import Path
import torch

VEHICLE_CLASSES = {"aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class DetectionConfig:
    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    output_dir: Path = PROJECT_ROOT / "outputs"

    image_size: int = 128
    grid_size: int = 16
    anchor_scales: tuple = (0.25, 0.5, 0.8)
    anchor_ratios: tuple = (0.5, 1.0, 2.0)

    positive_iou: float = 0.5
    negative_iou: float = 0.4
    neg_pos_ratio: int = 3

    score_threshold: float = 0.5
    nms_iou: float = 0.35

    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_train_images: int = 5000
    val_split: float = 0.1
    seed: int = 42

    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)

    device: torch.device = field(default_factory=get_device)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.output_dir = Path(self.output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def num_anchors_per_cell(self) -> int:
        return len(self.anchor_scales) * len(self.anchor_ratios)
