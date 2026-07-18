from __future__ import annotations
import math
import torch
from .config import DetectionConfig


def generate_anchors(cfg: DetectionConfig) -> torch.Tensor:
    cell = 1.0 / cfg.grid_size
    centers = (torch.arange(cfg.grid_size, dtype=torch.float32) + 0.5) * cell
    cy, cx = torch.meshgrid(centers, centers, indexing="ij")

    shapes = []
    for scale in cfg.anchor_scales:
        for ratio in cfg.anchor_ratios:
            w = scale * math.sqrt(ratio)
            h = scale / math.sqrt(ratio)
            shapes.append((w, h))
    shapes = torch.tensor(shapes, dtype=torch.float32)

    cx = cx.reshape(-1, 1).expand(-1, len(shapes)).reshape(-1)
    cy = cy.reshape(-1, 1).expand(-1, len(shapes)).reshape(-1)
    wh = shapes.repeat(cfg.grid_size * cfg.grid_size, 1)

    return torch.stack([cx, cy, wh[:, 0], wh[:, 1]], dim=1)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-8)


def encode_boxes(gt_cxcywh: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    tx = (gt_cxcywh[:, 0] - anchors[:, 0]) / anchors[:, 2]
    ty = (gt_cxcywh[:, 1] - anchors[:, 1]) / anchors[:, 3]
    tw = torch.log(gt_cxcywh[:, 2] / anchors[:, 2] + 1e-8)
    th = torch.log(gt_cxcywh[:, 3] / anchors[:, 3] + 1e-8)
    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(offsets: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    cx = offsets[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = offsets[:, 1] * anchors[:, 3] + anchors[:, 1]
    w = torch.exp(offsets[:, 2].clamp(max=4.0)) * anchors[:, 2]
    h = torch.exp(offsets[:, 3].clamp(max=4.0)) * anchors[:, 3]
    boxes = cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=1))
    return boxes.clamp(0.0, 1.0)


def match_anchors(
    gt_xyxy: torch.Tensor, anchors: torch.Tensor, cfg: DetectionConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    num_anchors = anchors.size(0)
    labels = torch.zeros(num_anchors, dtype=torch.float32)
    matched_gt = torch.zeros(num_anchors, dtype=torch.long)

    if gt_xyxy.numel() == 0:
        return labels, matched_gt

    ious = box_iou(cxcywh_to_xyxy(anchors), gt_xyxy)
    best_gt_iou, best_gt_idx = ious.max(dim=1)

    labels[best_gt_iou >= cfg.positive_iou] = 1.0
    ignore = (best_gt_iou >= cfg.negative_iou) & (best_gt_iou < cfg.positive_iou)
    labels[ignore] = -1.0

    best_anchor_idx = ious.argmax(dim=0)
    labels[best_anchor_idx] = 1.0
    matched_gt[:] = best_gt_idx
    matched_gt[best_anchor_idx] = torch.arange(gt_xyxy.size(0))

    return labels, matched_gt
