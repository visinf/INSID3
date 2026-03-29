"""Chest X-ray few-shot semantic segmentation dataset."""
from __future__ import annotations

import os
import glob
from os.path import join
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from utils.data import build_transform


class DatasetLung(Dataset):
    def __init__(self, datapath: str, transform, shot: int, num: int = 600):
        self.split = 'test'
        self.nclass = 1
        self.benchmark = 'lung'
        self.shot = shot
        self.num = num

        self.base_path = join(datapath, 'LungSegmentation')
        self.img_path = join(self.base_path, 'CXR_png')
        self.ann_path = join(self.base_path, 'masks')
        self.transform = transform

        self.categories = ['1']

        self.class_ids = range(0, 1)
        self.img_metadata_classwise = self.build_img_metadata_classwise()       

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int) -> dict:
        tgt_name, ref_names, class_sample = self.sample_episode(idx)
        tgt_img, tgt_mask, ref_imgs, ref_masks = self.load_frame(tgt_name, ref_names)
        tgt_img = self.transform(tgt_img)
        tgt_mask = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0).float(), tgt_img.size()[-2:], mode='nearest').squeeze()
        ref_imgs = torch.stack([self.transform(ref_img) for ref_img in ref_imgs])
        ref_masks_tmp = []
        for smask in ref_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), ref_imgs.size()[-2:], mode='nearest').squeeze()
            ref_masks_tmp.append(smask)
        ref_masks = torch.stack(ref_masks_tmp)

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask,
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, tgt_name: str, ref_names: list[str]) -> tuple:
        tgt_mask = self.read_mask(tgt_name)
        ref_masks = [self.read_mask(name) for name in ref_names]
        if tgt_name.find('MCUCXR')!=-1:
            tgt_id = tgt_name
        else:
            tgt_id = tgt_name[:-9] + '.png'

        tgt_img = Image.open(os.path.join(self.img_path, os.path.basename(tgt_id))).convert('RGB')
        ref_ids = []
        for name in ref_names:
            if name.find('MCUCXR')!=-1:
                ref_ids.append(os.path.basename(name))
            else:
                ref_ids.append(os.path.basename(name)[:-9] + '.png')

        ref_names = [os.path.join(self.img_path, sid) for sid in ref_ids]
        ref_imgs = [Image.open(name).convert('RGB') for name in ref_names]

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

        for cat in self.categories:
            img_paths = sorted([path for path in glob.glob('%s/*' % self.ann_path)])
            for img_path in img_paths:
                if os.path.basename(img_path).split('.')[1] == 'png':
                    img_metadata_classwise[cat] += [img_path]
        return img_metadata_classwise
    

def build(args) -> DatasetLung:
    transform = build_transform(args.image_size)
    dataset = DatasetLung(datapath=args.data_root, transform=transform,
                 shot=args.shots)
    return dataset
