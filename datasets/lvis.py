"""LVIS-92i few-shot semantic segmentation dataset."""
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from utils.data import polygons_to_bitmask, build_transform
import pycocotools.mask as mask_util


class DatasetLVIS(Dataset):
    def __init__(self, datapath: str, fold: int, transform, shot: int):
        self.fold = fold
        self.nfolds = 10
        self.benchmark = 'lvis'
        self.shot = shot
        self.anno_path = os.path.join(datapath, "LVIS")
        self.base_path = os.path.join(datapath, "LVIS", 'coco')
        self.transform = transform

        self.nclass, self.class_ids_ori, self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))

        self.img_metadata = self.build_img_metadata()

    def __len__(self) -> int:
        return 2300

    def __getitem__(self, idx: int) -> dict:
        idx %= len(self.class_ids)

        tgt_img, tgt_mask, ref_imgs, ref_masks, tgt_name, ref_names, class_sample, org_tgt_imsize = self.load_frame(idx)

        tgt_img = self.transform(tgt_img)
        tgt_mask = F.interpolate(tgt_mask.unsqueeze(0).unsqueeze(0).float(), tgt_img.size()[-2:], mode='nearest').squeeze()

        ref_imgs = torch.stack([self.transform(ref_img) for ref_img in ref_imgs])
        for midx, smask in enumerate(ref_masks):
            ref_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), ref_imgs.size()[-2:], mode='nearest').squeeze()
        ref_masks = torch.stack(ref_masks)

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask,
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(self.class_ids_c[class_sample])
        }
            
        return batch

    def build_img_metadata_classwise(self) -> dict:
        with open(os.path.join(self.anno_path, 'lvis_val.pkl'), 'rb') as f:
            val_anno = pickle.load(f)

        val_cat_ids = [i for i in list(val_anno.keys()) if len(val_anno[i]) > self.shot]
        nclass = len(val_cat_ids)
        nclass_per_fold = nclass // self.nfolds

        if self.fold != -1:
            class_ids = [val_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_per_fold)]
        else:
            class_ids = [val_cat_ids[self.nfolds * v] for v in range(nclass_per_fold)]

        return nclass, class_ids, val_anno

    def build_img_metadata(self) -> list:
        img_metadata = []
        for k in self.img_metadata_classwise.keys():
            img_metadata.extend(list(self.img_metadata_classwise[k].keys()))
        return sorted(list(set(img_metadata)))

    def get_mask(self, segm, image_size: tuple) -> torch.Tensor:

        if isinstance(segm, list):
            # polygon
            # polygons = [np.asarray(p).reshape(-1, 2)[:,::-1] for p in segm]
            # polygons = [p.reshape(-1) for p in polygons]
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self, idx: int) -> tuple:
        class_sample = self.class_ids_ori[idx]
        tgt_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
        tgt_info = self.img_metadata_classwise[class_sample][tgt_name]
        tgt_img = Image.open(os.path.join(self.base_path, tgt_name)).convert('RGB')
        org_tgt_imsize = tgt_img.size
        tgt_annos = tgt_info['annotations']
        segms = []

        for anno in tgt_annos:
            segms.append(self.get_mask(anno['segmentation'], org_tgt_imsize)[None, ...].float())
        tgt_mask = torch.cat(segms, dim=0)
        tgt_mask = tgt_mask.sum(0) > 0

        ref_names = []
        ref_pre_masks = []
        while True:  # keep sampling until reference != target
            ref_name = np.random.choice(list(self.img_metadata_classwise[class_sample].keys()), 1, replace=False)[0]
            if tgt_name != ref_name:
                ref_names.append(ref_name)
                ref_info = self.img_metadata_classwise[class_sample][ref_name]
                ref_annos = ref_info['annotations']

                ref_segms = []
                for anno in ref_annos:
                    ref_segms.append(anno['segmentation'])
                ref_pre_masks.append(ref_segms)

            if len(ref_names) == self.shot:
                break


        ref_imgs = []
        ref_masks = []
        for ref_name, ref_pre_mask in zip(ref_names, ref_pre_masks):
            ref_img = Image.open(os.path.join(self.base_path, ref_name)).convert('RGB')
            ref_imgs.append(ref_img)
            org_sup_imsize = ref_img.size
            sup_masks = []
            for pre_mask in ref_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...].float())
            ref_mask = torch.cat(sup_masks, dim=0)
            ref_mask = ref_mask.sum(0) > 0

            ref_masks.append(ref_mask)

        return tgt_img, tgt_mask, ref_imgs, ref_masks, tgt_name, ref_names, class_sample, org_tgt_imsize


def build(args) -> DatasetLVIS:
    transform = build_transform(args.image_size)
    dataset = DatasetLVIS(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots)
    return dataset
