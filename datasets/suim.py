"""SUIM few-shot semantic segmentation dataset."""
from __future__ import annotations

import os
import glob

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np


class DatasetSUIM(Dataset):
    def __init__(self, datapath: str, shot: int):
        self.benchmark = 'suim'
        self.shot = shot

        self.base_path = datapath
        self.img_path = os.path.join(self.base_path, 'images')
        self.ann_path = os.path.join(self.base_path, 'masks')

        self.categories = ['FV','HD','PF','RI','RO','SR','WR']

        self.class_ids = range(len(self.categories))
        self.img_metadata_classwise, self.num_images = self.build_img_metadata_classwise()

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> dict:
        tgt_name, ref_names, class_sample = self.sample_episode(idx)
        tgt_img, tgt_mask, ref_imgs, ref_masks = self.load_frame(tgt_name, ref_names)

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask.float(),
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(class_sample)}

        return batch


    def load_frame(self, tgt_mask_path: str, ref_mask_paths: list[str]) -> tuple:
        def maskpath_to_imgpath(maskpath):
            filename, imgext = maskpath.split('/')[-1].split('.')[0], '.jpg'
            return os.path.join(self.img_path, filename) + imgext

        tgt_img = Image.open(maskpath_to_imgpath(tgt_mask_path)).convert('RGB')

        ref_imgs = [Image.open(maskpath_to_imgpath(s_mask_path)).convert('RGB') for s_mask_path in ref_mask_paths]

        tgt_mask = self.read_mask(tgt_mask_path)
        ref_masks = [self.read_mask(s_mask_path) for s_mask_path in ref_mask_paths]

        return tgt_img, tgt_mask, ref_imgs, ref_masks

    def read_mask(self, img_name: str) -> torch.Tensor:
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx: int) -> tuple:
        class_id = idx % len(self.class_ids)
        class_sample = self.categories[class_id]

        tgt_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        ref_names = []
        while True:  # keep sampling until reference != target
            ref_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if tgt_name != ref_name: ref_names.append(ref_name)
            if len(ref_names) == self.shot: break

        return tgt_name, ref_names, class_id


    def build_img_metadata_classwise(self) -> tuple:
        num_images=0
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        for cat in self.categories:
            mask_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(self.base_path, 'masks', cat))])
            for mask_path in mask_paths:
                if self.read_mask(mask_path).count_nonzero() > 0: #no empty masks
                    img_metadata_classwise[cat] += [mask_path]
                    num_images += 1
        return img_metadata_classwise, num_images


def build(args) -> DatasetSUIM:
    dataset = DatasetSUIM(datapath=os.path.join(args.data_root, 'SUIM'),
                          shot=args.shots)
    return dataset
