"""PACO-Part few-shot semantic segmentation dataset."""
import os
import pickle

from torch.utils.data import Dataset
import torch
import PIL.Image as Image
import numpy as np
from utils.data import polygons_to_bitmask
import pycocotools.mask as mask_util
from os.path import join


class DatasetPACOPart(Dataset):
    def __init__(self, datapath: str, fold: int, shot: int, box_crop: bool = True):
        self.fold = fold
        self.nfolds = 4
        self.nclass = 448
        self.benchmark = 'paco_part'
        self.shot = shot
        self.img_path =  join(datapath, 'PACO-Part', 'coco')
        self.anno_path = join(datapath, 'PACO-Part', 'paco')

        self.box_crop = box_crop

        self.class_ids_ori, self.cid2img, self.img2anno = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))
        self.img_metadata = self.build_img_metadata()

    def __len__(self) -> int:
        return 2500

    def __getitem__(self, idx: int) -> dict:
        ok = False
        while not ok:
            ok = True
            tgt_img, tgt_mask, ref_imgs, ref_masks, tgt_name, ref_names, class_sample, org_tgt_imsize = self.load_frame()
            for smask in ref_masks:
                if 0 in smask.shape:
                    ok = False
                    break
            if 0 in tgt_mask.shape:
                ok = False
            
        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask.float(),
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(self.class_ids_c[class_sample])
        }

        return batch

    def build_img_metadata_classwise(self) -> tuple:

        with open(join(self.anno_path, 'paco_part_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(join(self.anno_path, 'paco_part_val.pkl'), 'rb') as f:
            test_anno = pickle.load(f)

        # Remove Duplicates
        new_cid2img = {}

        for cid_id in test_anno['cid2img']:
            id_list = []
            if cid_id not in new_cid2img:
                new_cid2img[cid_id] = []
            for img in test_anno['cid2img'][cid_id]:
                img_id = list(img.keys())[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    new_cid2img[cid_id].append(img)
        test_anno['cid2img'] = new_cid2img

        train_cat_ids = list(train_anno['cid2img'].keys())
        test_cat_ids = [i for i in list(test_anno['cid2img'].keys()) if len(test_anno['cid2img'][i]) > self.shot]
        assert len(train_cat_ids) == self.nclass

        nclass_per_fold = self.nclass // self.nfolds

        if self.fold != -1:
            class_ids = [train_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_per_fold)]
        else:
            class_ids = [train_cat_ids[self.nfolds * v] for v in range(nclass_per_fold)]
        class_ids = [x for x in class_ids if x in test_cat_ids]

        cid2img = test_anno['cid2img']
        img2anno = test_anno['img2anno']

        return class_ids, cid2img, img2anno

    def build_img_metadata(self) -> list:
        img_metadata = []
        for k in self.cid2img.keys():
            img_metadata += self.cid2img[k]
        return img_metadata

    def get_mask(self, segm, image_size: tuple) -> torch.Tensor:

        if isinstance(segm, list):
            # polygon
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

    def load_frame(self) -> tuple:
        class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        tgt_sample = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
        tgt_id, tgt_name = list(tgt_sample.keys())[0], list(tgt_sample.values())[0]
        tgt_name = '/'.join( tgt_name.split('/')[-2:])
        tgt_img = Image.open(os.path.join(self.img_path, tgt_name)).convert('RGB')
        org_tgt_imsize = tgt_img.size
        tgt_annos = self.img2anno[tgt_id]

        tgt_obj_dict = {}

        for anno in tgt_annos:
            if anno['category_id'] == class_sample:
                obj_id = anno['obj_ann_id']
                if obj_id not in tgt_obj_dict:
                    tgt_obj_dict[obj_id] = {
                        'obj_bbox': [],
                        'segms': []
                    }
                tgt_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                tgt_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_tgt_imsize)[None, ...])

        sel_tgt_id = np.random.choice(list(tgt_obj_dict.keys()), 1, replace=False)[0]
        tgt_obj_bbox = tgt_obj_dict[sel_tgt_id]['obj_bbox'][0]
        tgt_part_masks = tgt_obj_dict[sel_tgt_id]['segms']
        tgt_mask = torch.cat(tgt_part_masks, dim=0)
        tgt_mask = tgt_mask.sum(0) > 0

        ref_names = []
        ref_pre_masks = []
        ref_boxes = []
        while True:  # keep sampling until reference != target
            ref_sample = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            ref_id, ref_name = list(ref_sample.keys())[0], list(ref_sample.values())[0]
            ref_name = '/'.join(ref_name.split('/')[-2:])
            if tgt_name != ref_name:
                ref_names.append(ref_name)
                ref_annos = self.img2anno[ref_id]

                ref_obj_dict = {}
                for anno in ref_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in ref_obj_dict:
                            ref_obj_dict[obj_id] = {
                                'obj_bbox': [],
                                'segms': []
                            }
                        ref_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        ref_obj_dict[obj_id]['segms'].append(anno['segmentation'])

                sel_ref_id = np.random.choice(list(ref_obj_dict.keys()), 1, replace=False)[0]
                ref_obj_bbox = ref_obj_dict[sel_ref_id]['obj_bbox'][0]
                ref_part_masks = ref_obj_dict[sel_ref_id]['segms']

                ref_boxes.append(ref_obj_bbox)
                ref_pre_masks.append(ref_part_masks)

            if len(ref_names) == self.shot:
                break

        ref_imgs = []
        ref_masks = []
        for ref_name, ref_pre_mask in zip(ref_names, ref_pre_masks):
            ref_img = Image.open(os.path.join(self.img_path, ref_name)).convert('RGB')
            ref_imgs.append(ref_img)
            org_sup_imsize = ref_img.size
            sup_masks = []
            for pre_mask in ref_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...])
            ref_mask = torch.cat(sup_masks, dim=0)
            ref_mask = ref_mask.sum(0) > 0

            ref_masks.append(ref_mask)

        if self.box_crop:
            tgt_img = np.asarray(tgt_img)
            tgt_img = tgt_img[int(tgt_obj_bbox[1]):int(tgt_obj_bbox[1]+tgt_obj_bbox[3]), int(tgt_obj_bbox[0]):int(tgt_obj_bbox[0]+tgt_obj_bbox[2])]
            tgt_img = Image.fromarray(np.uint8(tgt_img))
            org_tgt_imsize = tgt_img.size
            tgt_mask = tgt_mask[int(tgt_obj_bbox[1]):int(tgt_obj_bbox[1]+tgt_obj_bbox[3]), int(tgt_obj_bbox[0]):int(tgt_obj_bbox[0]+tgt_obj_bbox[2])]

            new_ref_imgs = []
            new_ref_masks = []

            for sup_img, sup_mask, sup_box in zip(ref_imgs, ref_masks, ref_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_ref_imgs.append(sup_img)
                new_ref_masks.append(sup_mask[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])])

            ref_imgs = new_ref_imgs
            ref_masks = new_ref_masks

        return tgt_img, tgt_mask, ref_imgs, ref_masks, tgt_name, ref_names, class_sample, org_tgt_imsize
    

def build(args) -> DatasetPACOPart:
    dataset = DatasetPACOPart(datapath=args.data_root, fold=args.fold, shot=args.shots)
    return dataset
