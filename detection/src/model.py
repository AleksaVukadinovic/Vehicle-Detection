from __future__ import annotations
import torch
import torch.nn as nn
from .config import DetectionConfig


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = True) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VehicleDetector(nn.Module):
    def __init__(self, cfg: DetectionConfig) -> None:
        super().__init__()
        self.num_anchors = cfg.num_anchors_per_cell
        self.backbone = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.objectness = nn.Conv2d(256, self.num_anchors, kernel_size=1)
        self.regression = nn.Conv2d(256, self.num_anchors * 4, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.objectness.bias, -4.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.head(self.backbone(x))
        batch = x.size(0)

        obj = self.objectness(features)
        obj = obj.permute(0, 2, 3, 1).reshape(batch, -1)

        reg = self.regression(features)
        reg = reg.permute(0, 2, 3, 1).reshape(batch, -1, 4)

        return obj, reg


def build_model(cfg: DetectionConfig) -> VehicleDetector:
    return VehicleDetector(cfg)
