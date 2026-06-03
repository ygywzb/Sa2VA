import argparse
import os
import random
import sys
from functools import partial

import numpy as np
import torch
from mmengine.config import Config
from xtuner.registry import BUILDER


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate dataset loading and collate outputs before forward."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default=os.path.join(
            "projects",
            "sa2va",
            "configs",
            "dev",
            "ablation",
            "sa2va_in25_1b_bgt30_wo_CAS.py",
        ),
        help="Path to config file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Max samples per dataset to check. -1 means all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for collate checks.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_collate_fn(collate_cfg):
    if collate_cfg is None:
        return None
    if callable(collate_cfg):
        return collate_cfg
    if isinstance(collate_cfg, dict):
        collate_type = collate_cfg.get("type", None)
        if callable(collate_type):
            kwargs = {k: v for k, v in collate_cfg.items() if k != "type"}
            if kwargs:
                return partial(collate_type, **kwargs)
            return collate_type
    raise TypeError(f"Unsupported collate_fn config: {collate_cfg}")


def iter_child_datasets(dataset):
    if hasattr(dataset, "datasets"):
        for child in dataset.datasets:
            yield child
    else:
        yield dataset


def select_indices(total_len: int, max_samples: int):
    if max_samples < 0 or max_samples >= total_len:
        return list(range(total_len))
    if max_samples == 0:
        return []
    return np.linspace(0, total_len - 1, max_samples, dtype=int).tolist()


def validate_sample(sample, dataset_name, index):
    if sample is None:
        raise RuntimeError(f"{dataset_name} index {index} returned None")
    if "input_ids" not in sample or "labels" not in sample:
        raise KeyError(f"{dataset_name} index {index} missing input_ids/labels")
    if len(sample["input_ids"]) == 0:
        raise ValueError(f"{dataset_name} index {index} has empty input_ids")
    if len(sample["input_ids"]) != len(sample["labels"]):
        raise ValueError(
            f"{dataset_name} index {index} input_ids/labels length mismatch"
        )


def validate_batch(batch):
    if not isinstance(batch, dict) or "data" not in batch:
        raise KeyError("Collate output missing 'data'")
    data = batch["data"]
    required = ["input_ids", "attention_mask", "position_ids", "labels"]
    for key in required:
        if key not in data:
            raise KeyError(f"Collate output missing '{key}'")
    if data["input_ids"].shape != data["attention_mask"].shape:
        raise ValueError("input_ids/attention_mask shape mismatch")
    if data["input_ids"].shape != data["labels"].shape:
        raise ValueError("input_ids/labels shape mismatch")


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)

    cfg = Config.fromfile(args.config)
    train_dataloader = cfg.get("train_dataloader", None)
    train_dataset = cfg.get("train_dataset", None)

    if train_dataloader is not None and "dataset" in train_dataloader:
        dataset_cfg = train_dataloader["dataset"]
        batch_size = train_dataloader.get("batch_size", 1)
        collate_cfg = train_dataloader.get("collate_fn", None)
    elif train_dataset is not None:
        dataset_cfg = train_dataset
        batch_size = getattr(cfg, "batch_size", 1)
        collate_cfg = None
    else:
        raise KeyError("Cannot find train_dataloader.dataset or train_dataset in config")

    if args.batch_size is not None:
        batch_size = args.batch_size

    dataset = BUILDER.build(dataset_cfg)
    collate_fn = build_collate_fn(collate_cfg)

    for child in iter_child_datasets(dataset):
        dataset_name = getattr(child, "name", child.__class__.__name__)
        total_len = child.real_len() if hasattr(child, "real_len") else len(child)
        indices = select_indices(total_len, args.max_samples)
        print(f"Checking {dataset_name}: {len(indices)} / {total_len} samples", flush=True)

        batch_samples = []
        for idx in indices:
            sample = child.prepare_data(idx)
            validate_sample(sample, dataset_name, idx)
            batch_samples.append(sample)

            if collate_fn is not None and len(batch_samples) >= batch_size:
                batch = collate_fn(batch_samples)
                validate_batch(batch)
                batch_samples = []

        if collate_fn is not None and batch_samples:
            batch = collate_fn(batch_samples)
            validate_batch(batch)

    print("Dataset loading validation passed.", flush=True)


if __name__ == "__main__":
    main()
