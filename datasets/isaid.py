"""iSAID-5i few-shot semantic segmentation dataset."""
from __future__ import annotations

import os

import torch
import PIL.Image as Image
import numpy as np
from utils.data import build_transform
from torch.utils.data import Dataset
import torch.nn.functional as F


class DatasetISAID(Dataset):
    def __init__(self, datapath: str, fold: int, transform, shot: int) -> None:
        self.split = 'val'
        self.fold = fold
        self.nfolds = 3
        self.nclass = 15
        self.benchmark = 'isaid'
        self.shot = shot

        self.datapath = datapath
        self.img_path = os.path.join(datapath, 'val/images')
        self.ann_path = os.path.join(datapath, 'val/semantic_png')

        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def build_class_ids(self) -> list[int]:
        nclass_per_fold = self.nclass // self.nfolds
        class_ids = [self.fold * nclass_per_fold + i for i in range(nclass_per_fold)]
        return class_ids

    def sample_episode(self, idx: int) -> tuple:
        tgt_name, class_sample = self.img_metadata[idx]

        ref_names = []
        while True:  # keep sampling until reference != target
            ref_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if tgt_name != ref_name: ref_names.append(ref_name)
            if len(ref_names) == self.shot: break

        return tgt_name, ref_names, class_sample

    def __len__(self) -> int:
        return len(self.img_metadata)

    def read_mask(self, img_name: str) -> torch.Tensor:
        mask = torch.tensor(np.array(Image.open(os.path.join(self.ann_path, img_name) + '_instance_color_RGB.png')))
        return mask

    def load_frame(self, tgt_name: str, ref_names: list[str]) -> tuple:
        tgt_img = self.read_img(tgt_name)
        tgt_mask = self.read_mask(tgt_name)
        ref_imgs = [self.read_img(name) for name in ref_names]
        ref_masks = [self.read_mask(name) for name in ref_names]

        org_tgt_imsize = tgt_img.size

        return tgt_img, tgt_mask, ref_imgs, ref_masks, org_tgt_imsize

    def read_img(self, img_name: str) -> Image.Image:
        return Image.open(os.path.join(self.img_path, img_name) + '.png')

    def extract_ignore_idx(self, mask: torch.Tensor, class_id: int) -> tuple:
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def __getitem__(self, idx: int) -> dict:
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        tgt_name, ref_names, class_sample = self.sample_episode(idx)
        tgt_img, tgt_cmask, ref_imgs, ref_cmasks, org_tgt_imsize = self.load_frame(tgt_name, ref_names)

        tgt_img = self.transform(tgt_img)
        tgt_cmask = F.interpolate(tgt_cmask.unsqueeze(0).unsqueeze(0).float(), tgt_img.size()[-2:], mode='nearest').squeeze()
        tgt_mask, tgt_ignore_idx = self.extract_ignore_idx(tgt_cmask.float(), class_sample)

        ref_imgs = torch.stack([self.transform(ref_img) for ref_img in ref_imgs])

        ref_masks = []
        ref_ignore_idxs = []
        for scmask in ref_cmasks:
            scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), ref_imgs.size()[-2:], mode='nearest').squeeze()
            ref_mask, ref_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            ref_masks.append(ref_mask)
            ref_ignore_idxs.append(ref_ignore_idx)
        ref_masks = torch.stack(ref_masks)
        ref_ignore_idxs = torch.stack(ref_ignore_idxs)

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask,
                 'tgt_ignore_idx': tgt_ignore_idx,
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def build_img_metadata(self) -> list:
        def read_metadata(split, fold_id):
            fold_n_metadata = os.path.join(f'{self.datapath}/splits/{split}/fold{fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = read_metadata(self.split, self.fold)

        return img_metadata

    def build_img_metadata_classwise(self) -> dict:
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise


def build(args) -> DatasetISAID:
    transform = build_transform(args.image_size)
    dataset = DatasetISAID(datapath=os.path.join(args.data_root, 'iSAID'), fold=args.fold,
                           transform=transform, shot=args.shots)
    return dataset