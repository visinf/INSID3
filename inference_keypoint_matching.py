"""Semantic correspondence inference."""

import argparse
import datetime
import os
import random
import sys
import time
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import opts
from datasets import build_dataset
from models import build_insid3_from_args
from utils.metrics import PCKEvaluator


def main(args: argparse.Namespace) -> tuple:
    print(args)

    # ──────── Reproducibility and logging setup ────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_file = join(args.output_dir, 'log.txt')
    with open(log_file, 'w') as fp:
        fp.write(" ".join(sys.argv) + '\n')
        fp.write(str(vars(args)) + '\n\n')

    # ──────── Model setup ────────
    model = build_insid3_from_args(args)
    model.to(args.device)
    model.eval()

    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print('Start inference')

    start_time = time.time()
    pck = evaluate(args, model, log_file)
    print(f'Total inference time: {time.time() - start_time:.1f}s')
    return pck


def evaluate(args: argparse.Namespace, model: torch.nn.Module, log_file: str) -> tuple:
    # ──────── Dataset and loader setup ────────
    ds = build_dataset("spair", args=args)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x[0],
    )
    evaluator = PCKEvaluator(pck_by='image')

    # ──────── Evaluation loop ────────
    pbar = tqdm(loader, ncols=80)
    for idx, batch in enumerate(pbar):
        src_img = batch['src_img']
        trg_img = batch['trg_img']
        src_kps = batch['src_kps'].to(args.device)
        trg_kps = batch['trg_kps'].to(args.device)

        model.set_reference(src_img)
        model.set_target(trg_img)
        pred_kps = model.match(src_kps, use_debiased=args.debiased)

        evaluator.calculate_pck(
            trg_kps.unsqueeze(0),
            pred_kps.unsqueeze(0),
            torch.tensor([batch['n_pts']]),
            [batch['category']],
            torch.tensor([batch['pck_thresh']]),
        )

        if (idx + 1) % 50 == 0:
            pck = evaluator.get_result()
            pbar.set_description(
                f'PCK@(0.05,0.1,0.15,0.2): '
                f'{pck[0]*100:.2f}, {pck[1]*100:.2f}, '
                f'{pck[2]*100:.2f}, {pck[3]*100:.2f}'
            )

    # ──────── Final results ────────
    pck = evaluator.get_result()
    if os.path.isdir(args.output_dir):
        evaluator.save_result(join(args.output_dir, 'per_class_score.txt'))

    out_str = (
        f'PCK@0.05 = {pck[0] * 100:.2f}, '
        f'PCK@0.10 = {pck[1] * 100:.2f}, '
        f'PCK@0.15 = {pck[2] * 100:.2f}, '
        f'PCK@0.20 = {pck[3] * 100:.2f}'
    )
    print(out_str)
    with open(log_file, 'a') as fp:
        fp.write(out_str + '\n')
    return pck


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'INSID3 inference on semantic correspondence',
        parents=[opts.get_args_parser()],
    )
    parser.add_argument('--debiased', action='store_true', help='Use positional debiased features')
    args = parser.parse_args()
    timestamp = datetime.datetime.now().strftime('%m%d_%H%M')
    args.output_dir = join(args.output_dir, f'{args.exp_name}_{timestamp}')
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
