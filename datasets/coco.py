"""COCO-20i few-shot semantic segmentation dataset."""
from __future__ import annotations

import os
import pickle
from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np
from os.path import join


class DatasetCOCO(Dataset):
    def __init__(self, datapath: str, fold: int, shot: int):
        self.split = 'val'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 80
        self.test_episodes = 1000
        self.benchmark = 'coco'
        self.shot = shot
        self.split_coco = 'train2014'
        self.base_path = join(datapath, 'COCO2014')

        self.class_ids = self.build_class_ids()
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()
            
    def __len__(self) -> int:
        return self.test_episodes

    def __getitem__(self, idx: int) -> dict:
        tgt_img, tgt_mask, ref_imgs, ref_masks, class_sample = self.load_frame()

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask.float(),
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(class_sample)
        }
        return batch

    def build_class_ids(self) -> list[int]:
        nclass_per_fold = self.nclass // self.nfolds
        if self.fold != -1:
            class_ids = [self.fold + self.nfolds * v for v in range(nclass_per_fold)]
        else:
            class_ids = list(range(self.nclass))
        return class_ids

    @staticmethod
    def load_pickle(pickle_path: str) -> dict:
        with open(pickle_path, 'rb') as fp:
            meta = pickle.load(fp)
        return meta

    def build_img_metadata_classwise(self) -> dict:
        if self.fold != -1:
            with open(f'{self.base_path}/splits/{self.split}/fold{self.fold}.pkl', 'rb') as f:
                img_metadata_classwise = pickle.load(f)
        else:
            split_meta_path = [f'{self.base_path}/splits/{self.split}/fold{str(idx)}.pkl' for idx in range(4)]
            split_meta = [self.load_pickle(meta_path) for meta_path in split_meta_path]
            img_metadata_classwise = {k: [] for k, v in split_meta[0].items()}
            for meta in split_meta:
                for class_id in meta:
                    if len(img_metadata_classwise[class_id]) == 0 and len(meta[class_id]) > 0:
                        img_metadata_classwise[class_id] = meta[class_id]
        return img_metadata_classwise

    def build_img_metadata(self) -> list:
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata += self.img_metadata_classwise[k]
        return sorted(list(set(img_metadata)))

    def read_mask(self, name: str) -> torch.Tensor:
        mask_path = os.path.join(self.base_path, 'annotations', name)
        mask = torch.tensor(np.array(Image.open(mask_path[:mask_path.index('.jpg')] + '.png')))
        return mask

    def load_frame(self) -> tuple:
        class_sample = np.random.choice(self.class_ids, 1, replace=False)[0]
        tgt_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
        tgt_img = Image.open(os.path.join(self.base_path, tgt_name)).convert('RGB')
        tgt_mask = self.read_mask(tgt_name)

        tgt_mask[tgt_mask != class_sample + 1] = 0
        tgt_mask[tgt_mask == class_sample + 1] = 1

        ref_names = []
        while True:
            ref_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if tgt_name != ref_name: ref_names.append(ref_name)
            if len(ref_names) == self.shot: break

        ref_imgs, ref_masks = [], []
        for ref_name in ref_names:
            ref_imgs.append(Image.open(os.path.join(self.base_path, ref_name)).convert('RGB'))
            ref_mask = self.read_mask(ref_name)
            ref_mask[ref_mask != class_sample + 1] = 0
            ref_mask[ref_mask == class_sample + 1] = 1
            ref_masks.append(ref_mask)

        return tgt_img, tgt_mask, ref_imgs, ref_masks, class_sample


def build(args) -> DatasetCOCO:
    dataset = DatasetCOCO(datapath=args.data_root, fold=args.fold, shot=args.shots)
    return dataset
