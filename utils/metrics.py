"""Segmentation and semantic correspondence metrics."""

from __future__ import annotations

import copy
import numpy as np
import torch


# ──────── evaluator for in-context segmentation ────────

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

# ──────── evaluator for semantic correspondence ────────

class PCKEvaluator:
    """Percentage of Correct Keypoints (PCK) evaluator."""

    def __init__(self, pck_by: str = 'image', avg_by: str = 'all'):
        self.alpha = [0.05, 0.1, 0.15, 0.2]
        self.by = pck_by
        self.avg_by = avg_by
        self.result = {}
        for alpha in self.alpha:
            self.result[f'pck{alpha}'] = {'all': []}

    def clear_result(self):
        self.result = {key: {'all': []} for key in self.result}

    def state_dict(self):
        return copy.deepcopy(self.result)

    def load_state_dict(self, state_dict):
        self.result = copy.deepcopy(state_dict)

    def merge_state_dict(self, state_dict):
        for alpha_key, alpha_result in state_dict.items():
            if alpha_key not in self.result:
                self.result[alpha_key] = {'all': []}
            for category, values in alpha_result.items():
                if category not in self.result[alpha_key]:
                    self.result[alpha_key][category] = []
                self.result[alpha_key][category].extend(values)

    def calculate_pck(
        self,
        trg_kps: torch.Tensor,
        matches: torch.Tensor,
        n_pts: torch.Tensor,
        categories,
        pckthres: torch.Tensor,
    ):
        B = trg_kps.shape[0]

        for b in range(B):
            npt = int(n_pts[b].item())
            thres = pckthres[b].item()
            category = categories[b]
            diff = torch.norm(trg_kps[b, :npt] - matches[b, :npt], dim=-1)
            for alpha in self.alpha:
                key = f'pck{alpha}'
                if category not in self.result[key]:
                    self.result[key][category] = []

                if self.by == 'image':
                    pck = (diff <= alpha * thres).float().mean().item()
                    self.result[key][category].append(pck)
                    self.result[key]['all'].append(pck)
                elif self.by == 'point':
                    pck = (diff <= alpha * thres).float().tolist()
                    self.result[key][category].extend(pck)
                    self.result[key]['all'].extend(pck)
                else:
                    raise ValueError("pck_by must be 'image' or 'point'")

    def avg_result_all(self):
        out = {}
        for alpha in self.alpha:
            out[f'pck{alpha}'] = {}
            for k, v in self.result[f'pck{alpha}'].items():
                out[f'pck{alpha}'][k] = np.array(v).mean() if len(v) > 0 else 0.0
        return out

    def get_result(self):
        result = self.avg_result_all()
        if self.avg_by == 'all':
            return tuple(result[f'pck{a}']['all'] for a in self.alpha)

        cat_list = [cat for cat in self.result[f'pck{self.alpha[0]}'] if cat != 'all']
        per_cat_res = {}
        for alpha in self.alpha:
            per_cat_avg = np.array([result[f'pck{alpha}'][cat] for cat in cat_list])
            per_cat_res[f'pck{alpha}'] = per_cat_avg.mean() if len(cat_list) > 0 else 0.0
        return tuple(per_cat_res[f'pck{a}'] for a in self.alpha)

    def print_summarize_result(self):
        result = self.avg_result_all()
        print(' ' * 16 + ''.join([f"{alpha:<10}" for alpha in self.alpha]))
        pcks = [f"{result[f'pck{alpha}']['all']:.4f}" for alpha in self.alpha]
        print(' ' * 12 + ''.join([f"{pck:<10}" for pck in pcks]))

    def save_result(self, save_file):
        result = self.avg_result_all()
        outstring = "\n"
        catstring = ""
        for alpha in self.alpha:
            cat_list = []
            pck_list = []
            for k, v in result[f'pck{alpha}'].items():
                if k != 'all':
                    cat_list.append(k)
                    pck_list.append(v)
            cat_list = np.array(cat_list)
            pck_list = np.array(pck_list)
            indices = np.argsort(cat_list)
            cat_list = cat_list[indices].tolist()
            pck_list = pck_list[indices].tolist()
            pck_list = [f"{pck:.2%}" for pck in pck_list]
            cat_list.append('all')
            pck_list.append(f"{result[f'pck{alpha}']['all']:.2%}")

            if len(catstring) == 0:
                catstring += ' ' * 12 + ''.join([f"{category:<12}" for category in cat_list]) + '\n'
                outstring += catstring
            row = f"{alpha:<12}" + ''.join([f"{pck:<12}" for pck in pck_list]) + '\n'
            outstring += row

        outstring += '-----------------------------------------------------------------\n'
        with open(save_file, 'w') as f:
            f.write(outstring)
