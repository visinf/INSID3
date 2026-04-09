"""PASCAL-Part few-shot semantic segmentation dataset."""
import os
from os.path import join
from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np
import json
import pycocotools.mask as mask_util


class DatasetPASCALPart(Dataset):
    def __init__(self, datapath: str, fold: int, shot: int, box_crop: bool = True):
        self.fold = fold
        self.benchmark = 'pascal_part'
        self.shot = shot
        self.box_crop = box_crop

        self.json_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json')
        self.img_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/JPEGImages/{}.jpg')
        self.anno_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes/{}.json')
        js = json.load(open(self.json_file, 'r'))

        # Select category (4 folds: animals, indoor, person, vehicles)
        self.cat = ['animals', 'indoor', 'person', 'vehicles'][fold]

        self.cat_annos = js[self.cat]

        cat_part_name = []
        cat_part_id = []
        new_id = 0
        for i, obj in enumerate(self.cat_annos['object']):
            for part in self.cat_annos['object'][obj]['part']:
                if len(self.cat_annos['object'][obj]['part'][part]['train']) > 0 and \
                        len(self.cat_annos['object'][obj]['part'][part]['val']) > 0:
                    if obj + '+' + part == 'aeroplane+TAIL':
                        continue
                    cat_part_name.append(obj + '+' + part)
                    cat_part_id.append(new_id)
                    new_id += 1

        self.cat_part_name = cat_part_name
        self.class_ids = self.cat_part_id = cat_part_id
        self.nclass = len(cat_part_id)

        self.img_metadata = self.build_img_metadata()

    def __len__(self) -> int:
        return min(len(self.img_metadata), 2500)

    def build_img_metadata(self) -> list:
        img_metadata = []
        for obj in self.cat_annos['object']:
            for part in self.cat_annos['object'][obj]['part']:
                img_metadata.extend(self.cat_annos['object'][obj]['part'][part]['val'])
        return img_metadata

    def __getitem__(self, idx: int) -> dict:
        idx %= len(self.class_ids)
        tgt_img, tgt_mask, ref_imgs, ref_masks, class_sample = self.sample_episode(idx)

        tgt_mask = torch.from_numpy(tgt_mask).float()
        ref_masks = [torch.from_numpy(smask).float() for smask in ref_masks]

        batch = {
            'tgt_img': tgt_img,
            'tgt_mask': tgt_mask,
            'ref_imgs': ref_imgs,
            'ref_masks': ref_masks,
            'class_id': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)])
        }

        return batch

    def sample_episode(self, idx: int):
        class_sample = self.cat_part_name[idx]
        obj_n, part_n = class_sample.split('+')

        # Query selection
        while True:
            query_img_id = np.random.choice(self.cat_annos['object'][obj_n]['part'][part_n]['val'], 1, replace=False)[0]
            anno = json.load(open(self.anno_file.format(query_img_id), 'r'))

            sel_obj_in_img = [o for o in anno['object'] if o['name'] == obj_n]
            if len(sel_obj_in_img) == 0:
                continue
            sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]

            sel_parts = [p for p in sel_obj['parts'] if p['name'] == part_n]
            if not sel_parts:
                continue

            part_masks = []
            for sel_part in sel_parts:
                part_masks.extend(sel_part['mask'])
            for mask in part_masks:
                mask['counts'] = mask['counts'].encode("ascii")
            part_mask = mask_util.decode(part_masks)
            part_mask = part_mask.sum(-1) > 0
            if part_mask.size > 0:
                break

        tgt_img = Image.open(self.img_file.format(query_img_id)).convert('RGB')
        org_tgt_imsize = tgt_img.size
        tgt_mask = part_mask
        query_obj_box = [int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']]

        # Support selection
        support_img_ids = []
        support_masks = []
        support_boxes = []

        while len(support_img_ids) < self.shot:
            support_img_id = np.random.choice(self.cat_annos['object'][obj_n]['part'][part_n]['val'], 1, replace=False)[0]
            if support_img_id == query_img_id or support_img_id in support_img_ids:
                continue

            anno = json.load(open(self.anno_file.format(support_img_id), 'r'))
            sel_obj_in_img = [o for o in anno['object'] if o['name'] == obj_n]
            if len(sel_obj_in_img) == 0:
                continue
            sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]
            sel_parts = [p for p in sel_obj['parts'] if p['name'] == part_n]
            if not sel_parts:
                continue

            part_masks = []
            for sel_part in sel_parts:
                part_masks.extend(sel_part['mask'])
            for mask in part_masks:
                mask['counts'] = mask['counts'].encode("ascii")
            part_mask = mask_util.decode(part_masks)
            part_mask = part_mask.sum(-1) > 0
            if part_mask.size == 0:
                continue

            support_img_ids.append(support_img_id)
            support_masks.append(part_mask)
            support_boxes.append([int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']])

        ref_imgs = [Image.open(self.img_file.format(sid)).convert('RGB') for sid in support_img_ids]

        if self.box_crop:
            tgt_img = np.asarray(tgt_img)
            tgt_img = tgt_img[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]
            tgt_img = Image.fromarray(np.uint8(tgt_img))
            org_tgt_imsize = tgt_img.size
            tgt_mask = tgt_mask[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]

            new_ref_imgs = []
            new_ref_masks = []
            for sup_img, sup_mask, sup_box in zip(ref_imgs, support_masks, support_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]]
                sup_img = Image.fromarray(np.uint8(sup_img))
                new_ref_imgs.append(sup_img)
                new_ref_masks.append(sup_mask[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]])
            ref_imgs = new_ref_imgs
            support_masks = new_ref_masks

        return tgt_img, tgt_mask, ref_imgs, support_masks, class_sample


def build(args) -> DatasetPASCALPart:
    dataset = DatasetPASCALPart(datapath=args.data_root, fold=args.fold, shot=args.shots)
    return dataset
