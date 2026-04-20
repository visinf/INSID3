"""SPair-71k semantic correspondence dataset."""
from __future__ import annotations

import json
from os.path import join

import PIL.Image as Image
import torch
from torch.utils.data import Dataset


class DatasetSPair(Dataset):
    def __init__(self, datapath: str, split: str = 'test'):
        self.split = split
        self.benchmark = 'spair'

        self.base_path = join(datapath, 'SPair-71k')
        self.layout_path = join(self.base_path, 'Layout', 'large', f'{split}.txt')
        self.img_path = join(self.base_path, 'JPEGImages')
        self.ann_path = join(self.base_path, 'PairAnnotation', split)

        self.categories = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
        ]
        self.class_dict = {cat: i for i, cat in enumerate(self.categories)}
        self.class_ids = range(len(self.categories))

        self.pair_metadata = self.build_pair_metadata()

        self.src_imnames = []
        self.trg_imnames = []
        self.src_kps = []
        self.trg_kps = []
        self.trg_bbox = []
        self.cls = []

        self.build_annotations()

    def __len__(self) -> int:
        return len(self.pair_metadata)

    def __getitem__(self, idx: int) -> dict:
        src_img = self.load_image(self.src_imnames[idx], self.cls[idx])
        trg_img = self.load_image(self.trg_imnames[idx], self.cls[idx])

        src_kps = self.src_kps[idx].clone()
        trg_kps = self.trg_kps[idx].clone()
        trg_bbox = self.trg_bbox[idx].clone()
        n_pts = src_kps.shape[0]

        pck_thresh = torch.tensor(
            max(trg_bbox[2] - trg_bbox[0], trg_bbox[3] - trg_bbox[1]),
            dtype=torch.float32,
        )

        batch = {
            'src_img': src_img,
            'trg_img': trg_img,
            'src_kps': src_kps,
            'trg_kps': trg_kps,
            'pck_thresh': pck_thresh,
            'n_pts': n_pts,
            'category': self.cls[idx],
            'class_id': torch.tensor(self.class_dict[self.cls[idx]]),
        }
        return batch

    def build_pair_metadata(self) -> list[str]:
        with open(self.layout_path, 'r') as fp:
            pair_metadata = [line.strip() for line in fp if line.strip()]
        return pair_metadata

    def build_annotations(self) -> None:
        for pair_name in self.pair_metadata:
            ann = self.load_annotation(pair_name)

            self.src_imnames.append(pair_name.split('-')[1] + '.jpg')
            self.trg_imnames.append(pair_name.split('-')[2].split(':')[0] + '.jpg')
            self.src_kps.append(torch.tensor(ann['src_kps']).float())   # (N, 2)
            self.trg_kps.append(torch.tensor(ann['trg_kps']).float())   # (N, 2)
            self.trg_bbox.append(torch.tensor(ann['trg_bndbox']).float())
            self.cls.append(ann['category'])

    def load_annotation(self, pair_name: str) -> dict:
        ann_path = join(self.ann_path, f'{pair_name}.json')
        with open(ann_path, 'r') as fp:
            ann = json.load(fp)
        return ann

    def load_image(self, img_name: str, category: str) -> Image.Image:
        return Image.open(join(self.img_path, category, img_name)).convert('RGB')


def build(args) -> DatasetSPair:
    split = getattr(args, 'split', 'test')
    dataset = DatasetSPair(datapath=args.data_root, split=split)
    return dataset