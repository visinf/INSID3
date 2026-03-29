"""Segmentation metrics."""

from __future__ import annotations

import torch


NCLASS = {
    'pascal_voc': 20, 'coco': 80, 'paco_part': 448,
    'lvis': 1203, 'lung': 2, 'isaid': 15,
    'isic': 3, 'suim': 7, 'permis': 1,
}


class AverageMeter:
    """Accumulates per-class intersection and union for mIoU computation."""

    def __init__(self, dataset_name: str, class_ids: list[int]):
        self.class_ids_interest = torch.tensor(class_ids).cuda()
        self.nclass = NCLASS.get(dataset_name, max(class_ids) + 1)
        self.intersection_buf = torch.zeros(2, self.nclass).float().cuda()
        self.union_buf = torch.zeros(2, self.nclass).float().cuda()

    def update(self, inter_b: torch.Tensor, union_b: torch.Tensor, class_id: torch.Tensor) -> None:
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())

    def compute_iou(self) -> tuple[torch.Tensor, torch.Tensor]:
        ones = torch.ones_like(self.union_buf)
        iou = self.intersection_buf.float() / torch.max(
            torch.stack([self.union_buf, ones]), dim=0
        )[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean() * 100

        fb_iou = (
            self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1)
            / self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)
        ).mean() * 100

        return miou, fb_iou


class Evaluator:
    """Computes intersection and union between prediction and ground truth."""

    ignore_index = 255

    @classmethod
    def classify_prediction(cls, pred_mask: torch.Tensor, gt_mask: torch.Tensor, tgt_ignore_idx: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute intersection and union between prediction and ground truth.

        Args:
            pred_mask: (H, W) binary prediction.
            gt_mask: (H, W) ground truth.
            tgt_ignore_idx: optional (H, W) ignore mask.

        Returns:
            (area_inter, area_union) each of shape (2, 1).
        """
        pred = pred_mask.float().unsqueeze(0)
        gt = (gt_mask > 0).to(pred.device).float()
        gt = gt.unsqueeze(0)

        if tgt_ignore_idx is not None:
            tgt_ignore_idx = tgt_ignore_idx.to(gt.device)
            assert torch.logical_and(tgt_ignore_idx, gt).sum() == 0
            tgt_ignore_idx *= cls.ignore_index
            gt = gt + tgt_ignore_idx
            pred[gt == cls.ignore_index] = cls.ignore_index

        area_inter, area_pred, area_gt = [], [], []
        for _pred, _gt in zip(pred, gt):
            _inter = _pred[_pred == _gt]
            if _inter.size(0) == 0:
                _area_inter = torch.tensor([0, 0], device=_pred.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt, bins=2, min=0, max=1))

        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter
        return area_inter, area_union
