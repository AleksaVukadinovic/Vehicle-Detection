"""Central configuration for the vehicle detection project."""

from dataclasses import dataclass, field
from pathlib import Path

import torch

# CIFAR-10 class indices that correspond to vehicles.
# CIFAR-10 labels: 0=airplane, 1=automobile, 2=bird, 3=cat, 4=deer,
#                  5=dog, 6=frog, 7=horse, 8=ship, 9=truck
VEHICLE_CLASS_INDICES = {0, 1, 8, 9}  # airplane, automobile, ship, truck

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS (Apple) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    """Hyper-parameters and paths used for training."""

    data_dir: Path = PROJECT_ROOT / "data"
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"

    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    seed: int = 42

    # Normalization stats for CIFAR-10.
    mean: tuple = (0.4914, 0.4822, 0.4465)
    std: tuple = (0.2470, 0.2435, 0.2616)

    device: torch.device = field(default_factory=get_device)

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
