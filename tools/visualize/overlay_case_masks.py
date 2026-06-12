import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overlay GT/baseline/ours binary masks onto frames for one exported case directory."
    )
    parser.add_argument(
        "case_dir",
        help="Case directory like ./work_dirs/visualize/mevis_case_pack/<video_id>/exp_x",
    )
    parser.add_argument(
        "--gt_color",
        default="0,255,0",
        help="Overlay RGB color for GT masks, formatted as R,G,B. Default: 0,255,0",
    )
    parser.add_argument(
        "--baseline_color",
        default="255,0,0",
        help="Overlay RGB color for baseline masks, formatted as R,G,B. Default: 255,0,0",
    )
    parser.add_argument(
        "--ours_color",
        default="255,0,0",
        help="Overlay RGB color for our masks, formatted as R,G,B. Default: 255,0,0",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay alpha in [0, 1]. Default: 0.45",
    )
    return parser.parse_args()


def parse_color(color_str: str):
    parts = [int(x.strip()) for x in color_str.split(',')]
    if len(parts) != 3:
        raise ValueError(f"Invalid mask color: {color_str}. Expected R,G,B")
    if any(x < 0 or x > 255 for x in parts):
        raise ValueError(f"Invalid mask color: {color_str}. Each value must be in [0, 255]")
    return np.array(parts, dtype=np.float32)


def overlay_mask(frame: Image.Image, mask: Image.Image, color: np.ndarray, alpha: float):
    frame_np = np.array(frame.convert('RGB'), dtype=np.float32)
    mask_np = np.array(mask.convert('L')) > 0
    if mask_np.shape[:2] != frame_np.shape[:2]:
        raise ValueError(
            f"Mask/frame size mismatch: mask {mask_np.shape[:2]}, frame {frame_np.shape[:2]}"
        )

    out = frame_np.copy()
    out[mask_np] = frame_np[mask_np] * (1.0 - alpha) + color * alpha
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def process_mask_set(case_dir: Path, mask_dir_name: str, output_dir_name: str, color: np.ndarray, alpha: float):
    frames_dir = case_dir / 'frames'
    masks_dir = case_dir / mask_dir_name
    output_dir = case_dir / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob('*.png'))
    if not frame_paths:
        raise FileNotFoundError(f"No frame png files found in {frames_dir}")

    for frame_path in frame_paths:
        mask_path = masks_dir / frame_path.name
        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_path}")
        frame = Image.open(frame_path)
        mask = Image.open(mask_path)
        overlaid = overlay_mask(frame, mask, color, alpha)
        overlaid.save(output_dir / frame_path.name)


def main():
    args = parse_args()
    case_dir = Path(args.case_dir)
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    gt_color = parse_color(args.gt_color)
    baseline_color = parse_color(args.baseline_color)
    ours_color = parse_color(args.ours_color)
    alpha = args.alpha
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Invalid alpha: {alpha}. Expected value in [0, 1]")

    process_mask_set(case_dir, 'gt_masks', 'gt_overlay', gt_color, alpha)
    process_mask_set(case_dir, 'baseline_masks', 'baseline_overlay', baseline_color, alpha)
    process_mask_set(case_dir, 'ours_masks', 'ours_overlay', ours_color, alpha)

    print(f"Saved overlays to: {case_dir / 'gt_overlay'}")
    print(f"Saved overlays to: {case_dir / 'baseline_overlay'}")
    print(f"Saved overlays to: {case_dir / 'ours_overlay'}")


if __name__ == '__main__':
    main()
