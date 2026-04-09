"""ISIC few-shot semantic segmentation dataset."""
from __future__ import annotations

import os
import glob

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np


class DatasetISIC(Dataset):
    def __init__(self, datapath: str, shot: int, num: int = 600):
        self.benchmark = 'isic'
        self.shot = shot
        self.num = num
        self.categories = ['1','2','3']
            
        self.base_path = os.path.join(datapath, 'ISIC')
        self.img_path = os.path.join(self.base_path, 'ISIC2018_Task1-2_Training_Input')
        self.ann_path = os.path.join(self.base_path, 'ISIC2018_Task1_Training_GroundTruth')

        self.class_ids = range(0, 3)
        self.img_metadata_classwise = self.build_img_metadata_classwise()       

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> dict:
        tgt_name, ref_names, class_sample = self.sample_episode(idx)

        tgt_img, tgt_mask, ref_imgs, ref_masks = self.load_frame(tgt_name, ref_names)

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask.float(),
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, tgt_name: str, ref_names: list[str]) -> tuple:
        tgt_img = Image.open(tgt_name).convert('RGB')
        ref_imgs = [Image.open(name).convert('RGB') for name in ref_names]

        tgt_id = tgt_name.split('/')[-1].split('.')[0]
        tgt_name = os.path.join(self.ann_path, tgt_id) + '_segmentation.png'
        ref_ids = [name.split('/')[-1].split('.')[0] for name in ref_names]
        ref_names = [os.path.join(self.ann_path, sid) + '_segmentation.png' for name, sid in zip(ref_names, ref_ids)]

        tgt_mask = self.read_mask(tgt_name)
        ref_masks = [self.read_mask(name) for name in ref_names]

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

    def build_img_metadata_classwise(self) -> dict:
        img_metadata_classwise = {}
        for cat in self.categories:
            img_metadata_classwise[cat] = []

        build_path = self.img_path

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % os.path.join(build_path, cat))])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'jpg':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
    

def build(args) -> DatasetISIC:
    dataset = DatasetISIC(datapath=args.data_root, shot=args.shots)
    return dataset
