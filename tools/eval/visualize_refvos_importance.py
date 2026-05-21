import argparse
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib

from projects.sa2va.evaluation.dataset import RefVOSDataset


DATASETS_INFO = {
    "DAVIS": {
        "data_root": "data/video_datas/davis17/",
        "image_folder": "data/video_datas/davis17/valid/JPEGImages/",
        "expression_file": "data/video_datas/davis17/meta_expressions/valid/meta_expressions.json",
        "mask_file": "data/video_datas/davis17/valid/mask_dict.pkl",
    },
    "MEVIS": {
        "data_root": "data/video_datas/mevis/valid/",
        "image_folder": "data/video_datas/mevis/valid/JPEGImages",
        "expression_file": "data/video_datas/mevis/valid/meta_expressions.json",
        "mask_file": None,
    },
    "MEVIS_U": {
        "data_root": "data/video_datas/mevis/valid_u/",
        "image_folder": "data/video_datas/mevis/valid_u/JPEGImages",
        "expression_file": "data/video_datas/mevis/valid_u/meta_expressions.json",
        "mask_file": "data/video_datas/mevis/valid_u/mask_dict.json",
    },
    "MEVIS_T": {
        "data_root": "data/video_datas/mevis/test/",
        "image_folder": "data/video_datas/mevis/test/JPEGImages",
        "expression_file": "data/video_datas/mevis/test/meta_expressions_release.json",
        "mask_file": None,
    },
    "REFYTVOS": {
        "data_root": "data/video_datas/rvos/",
        "image_folder": "data/video_datas/rvos/valid/JPEGImages/",
        "expression_file": "data/video_datas/rvos/meta_expressions/valid/meta_expressions.json",
        "mask_file": None,
    },
    "REVOS": {
        "data_root": "data/video_datas/revos/",
        "image_folder": "data/video_datas/revos/",
        "expression_file": "data/video_datas/revos/meta_expressions_valid_.json",
        "mask_file": None,
    },
    "REF_SAV": {
        "data_root": "data/ref_sav_eval/",
        "image_folder": "data/ref_sav_eval/videos",
        "expression_file": "data/ref_sav_eval/meta_expressions_valid.json",
        "mask_file": "data/ref_sav_eval/mask_dict.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize importance maps saved from refvos visualize eval"
    )
    parser.add_argument("--npz_dir", required=True, help="Directory with importance_*.npz")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_INFO.keys(),
        default="MEVIS",
        help="Dataset name (for loading frames)",
    )
    parser.add_argument("--data_root", default="./data", help="Root directory for all datasets.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only process the first N npz files; -1 means all.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (defaults to npz_dir).",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha.")
    parser.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name.")
    parser.add_argument("--pmin", type=float, default=1.0, help="Lower percentile for normalization.")
    parser.add_argument("--pmax", type=float, default=99.0, help="Upper percentile for normalization.")
    parser.add_argument(
        "--skip_overlay",
        action="store_true",
        help="Only save colored heatmaps, skip overlay.",
    )
    args = parser.parse_args()
    return args


def _resolve_dataset_info(dataset_name, data_root):
    info = DATASETS_INFO[dataset_name].copy()
    for path_key, path_val in info.items():
        if path_val is not None and (
            "folder" in path_key or "file" in path_key or "root" in path_key
        ):
            info[path_key] = os.path.join(data_root, os.path.relpath(path_val, "./data"))
    return info


def _load_frames(dataset_info, index):
    dataset = RefVOSDataset(
        image_folder=dataset_info["image_folder"],
        expression_file=dataset_info["expression_file"],
        mask_file=dataset_info["mask_file"],
    )
    item = dataset[index]
    return item["images"]


def _normalize_map(m, pmin, pmax):
    vmin, vmax = np.percentile(m, pmin), np.percentile(m, pmax)
    scale = vmax - vmin if vmax > vmin else 1.0
    return np.clip((m - vmin) / scale, 0.0, 1.0)


def _colorize(m, cmap_name):
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    return (cmap(m)[:, :, :3] * 255).astype(np.uint8)


def _save_colored_and_overlay(
    maps, frames, base_name, colored_dir, overlay_dir, alpha, cmap, pmin, pmax, skip_overlay
):
    maps_norm = _normalize_map(maps, pmin, pmax)

    if maps_norm.ndim == 2:
        maps_norm = maps_norm[None, ...]

    for i, m_norm in enumerate(maps_norm):
        color = _colorize(m_norm, cmap)
        color_img = Image.fromarray(color)
        color_path = os.path.join(colored_dir, f"{base_name}_{i:02d}.png")
        color_img.save(color_path)

        if skip_overlay:
            continue

        if frames is not None and i < len(frames):
            frame = frames[i]
            if frame.size != color_img.size:
                color_img = color_img.resize(frame.size, Image.BILINEAR)
            overlay = Image.blend(frame.convert("RGB"), color_img, alpha=alpha)
            overlay_path = os.path.join(overlay_dir, f"{base_name}_{i:02d}.png")
            overlay.save(overlay_path)


def main():
    args = parse_args()

    npz_dir = Path(args.npz_dir)
    output_dir = Path(args.output_dir or args.npz_dir)
    colored_dir = output_dir / "colored"
    overlay_dir = output_dir / "overlay"
    colored_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = _resolve_dataset_info(args.dataset, args.data_root)

    npz_files = sorted(npz_dir.glob("importance_*.npz"))
    if args.max_samples > 0:
        npz_files = npz_files[: args.max_samples]

    index_re = re.compile(r"importance_(\d+)\.npz")

    for npz_path in npz_files:
        match = index_re.search(npz_path.name)
        if not match:
            continue
        index = int(match.group(1))

        with np.load(npz_path) as data:
            if "importance_maps" not in data:
                continue
            maps = data["importance_maps"]
            tile_map = data.get("importance_tile_map")
            thumb_map = data.get("importance_thumbnail_map")

        frames = _load_frames(dataset_info, index)
        base_name = npz_path.stem
        _save_colored_and_overlay(
            maps,
            frames,
            base_name,
            str(colored_dir),
            str(overlay_dir),
            args.alpha,
            args.cmap,
            args.pmin,
            args.pmax,
            args.skip_overlay,
        )

        if tile_map is not None:
            _save_colored_and_overlay(
                tile_map,
                frames,
                base_name + "_tile",
                str(colored_dir),
                str(overlay_dir),
                args.alpha,
                args.cmap,
                args.pmin,
                args.pmax,
                args.skip_overlay,
            )

        if thumb_map is not None:
            _save_colored_and_overlay(
                thumb_map,
                frames,
                base_name + "_thumb",
                str(colored_dir),
                str(overlay_dir),
                args.alpha,
                args.cmap,
                args.pmin,
                args.pmax,
                args.skip_overlay,
            )

    print(f"Saved colored maps to {colored_dir}")
    if not args.skip_overlay:
        print(f"Saved overlays to {overlay_dir}")


if __name__ == "__main__":
    main()
