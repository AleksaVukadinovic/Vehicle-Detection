from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchors import encode_boxes, match_anchors, xyxy_to_cxcywh
from .config import DetectionConfig


class DetectionLoss(nn.Module):
    def __init__(self, cfg: DetectionConfig, anchors: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("anchors", anchors)

    def forward(
        self,
        obj_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        gt_boxes: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = obj_logits.device
        anchors = self.anchors.to("cpu")

        cls_losses, reg_losses = [], []
        for i, gt in enumerate(gt_boxes):
            labels, matched_gt = match_anchors(gt, anchors, self.cfg)
            labels = labels.to(device)

            pos_mask = labels == 1.0
            neg_mask = labels == 0.0
            num_pos = int(pos_mask.sum().item())

            obj_loss = F.binary_cross_entropy_with_logits(
                obj_logits[i], (labels == 1.0).float(), reduction="none"
            )

            pos_loss = obj_loss[pos_mask].sum()
            neg_loss = obj_loss[neg_mask]
            num_neg = min(self.cfg.neg_pos_ratio * max(num_pos, 1), neg_loss.numel())
            neg_loss = neg_loss.topk(num_neg).values.sum()
            cls_losses.append((pos_loss + neg_loss) / max(num_pos, 1))

            if num_pos > 0:
                matched = gt[matched_gt[pos_mask.cpu()]].to(device)
                targets = encode_boxes(
                    xyxy_to_cxcywh(matched), self.anchors[pos_mask]
                )
                reg_losses.append(
                    F.smooth_l1_loss(reg_preds[i][pos_mask], targets, reduction="sum")
                    / num_pos
                )

        cls_loss = torch.stack(cls_losses).mean()
        reg_loss = (
            torch.stack(reg_losses).mean()
            if reg_losses
            else torch.zeros((), device=device)
        )
        return cls_loss + reg_loss, cls_loss, reg_loss
