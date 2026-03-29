"""Command-line arguments for INSID3 inference."""

import argparse

SUPPORTED_DATASETS = [
    "coco", "lvis", "pascal_part", "paco_part",
    "isaid", "isic", "lung", "suim", "permis",
]


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("INSID3 inference", add_help=False)

    # Model
    parser.add_argument(
        "--model-size",
        default="large",
        choices=["small", "base", "large"],
        help="DINOv3 backbone size",
    )
    parser.add_argument(
        "--image-size",
        default=1024,
        type=int,
        help="Input image resolution",
    )
    parser.add_argument(
        "--crf-mask-refinement",
        action="store_true",
        help="Enable CRF-based mask refinement.",
    )

    # Episode
    parser.add_argument(
        "--shots",
        default=1,
        type=int,
        help="Number of reference images (shots)",
    )

    # Hyperparameters
    parser.add_argument(
        "--svd-comps",
        default=500,
        type=int,
        help="Number of SVD components for positional debiasing",
    )
    parser.add_argument(
        "--tau",
        default=0.6,
        type=float,
        help="Clustering distance threshold",
    )
    parser.add_argument(
        "--merge-thresh",
        default=0.2,
        type=float,
        help="Cluster aggregation threshold",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="coco",
        choices=SUPPORTED_DATASETS,
        help="Dataset for evaluation",
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Root directory of datasets",
    )
    parser.add_argument(
        "--fold",
        default=0,
        type=int,
        help="Fold index: for COCO and LVIS",
    )

    # Runtime
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for logs and results",
    )
    parser.add_argument(
        "--exp-name",
        default="insid3-coco",
        help="Run name",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed",
    )
    parser.add_argument(
        "--num-workers",
        default=0,
        type=int,
        help="Number of data loading workers",
    )

    return parser