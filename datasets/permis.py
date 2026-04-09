"""PerMIS few-shot semantic segmentation dataset."""
import os

from torch.utils.data import Dataset
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image


class DatasetPerMis(Dataset):
    def __init__(self, datapath: str, shot: int):
        self.benchmark = 'permis'
        self.shot = shot

        self.episodes = []
        for vid_id in tqdm(os.listdir(datapath)):
            masks_np = np.load(f"{datapath}/{vid_id}/masks.npz.npy", allow_pickle=True)

            for f in range(3):
                gt_mask = torch.tensor(list(masks_np[f].values())[0]).int()
                frame_path = f"{datapath}/{vid_id}/{f}.jpg"
                full_img = Image.open(frame_path).convert("RGB")
                if f == 0:
                    first_episode = {"supp_img": [full_img], "supp_mask": [gt_mask]}
                    second_episode = {"supp_img": [full_img], "supp_mask": [gt_mask]}
                if f == 1:
                    first_episode["tgt_img"] = full_img
                    first_episode["tgt_mask"] = gt_mask
                if f == 2:
                    second_episode["tgt_img"] = full_img
                    second_episode["tgt_mask"] = gt_mask
            self.episodes.append(first_episode)
            self.episodes.append(second_episode)

            self.class_ids = [0]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> dict:
        episode = self.episodes[idx]
        ref_imgs, ref_masks = episode["supp_img"], episode["supp_mask"]
        tgt_img, tgt_mask = episode["tgt_img"], episode["tgt_mask"]

        batch = {'tgt_img': tgt_img,
                 'tgt_mask': tgt_mask.float(),
                 'ref_imgs': ref_imgs,
                 'ref_masks': ref_masks,
                 'class_id': torch.tensor(0)
                 }

        return batch



def build(args) -> DatasetPerMis:
    dataset = DatasetPerMis(datapath=os.path.join(args.data_root, 'PerMIRS'),
                            shot=args.shots)
    return dataset
