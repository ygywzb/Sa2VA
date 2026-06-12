import argparse
import csv
import json
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as cocomask


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ReVOS qualitative cases with frames, GT masks, and prediction masks."
    )
    parser.add_argument(
        "--ours_eval_root",
        required=True,
        help="Root eval directory for our model, e.g. work_dirs/visualize/bgt40_eval.",
    )
    parser.add_argument(
        "--baseline_eval_root",
        required=True,
        help="Root eval directory for baseline model, e.g. work_dirs/visualize/eval.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case to export in the form video_id:exp_id. Can be specified multiple times.",
    )
    parser.add_argument(
        "--data_root",
        default="./data",
        help="Root directory for all datasets, e.g. data/baseline.",
    )
    parser.add_argument(
        "--exp_path",
        default="data/video_datas/revos/meta_expressions_valid_.json",
        help="ReVOS expression metadata path.",
    )
    parser.add_argument(
        "--mask_path",
        default="data/video_datas/revos/mask_dict.json",
        help="ReVOS GT mask_dict path.",
    )
    parser.add_argument(
        "--image_root",
        default="data/video_datas/revos",
        help="ReVOS frame root path.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Base output directory. Results will be saved to output_dir/visualize/revos_case_pack_manual.",
    )
    return parser.parse_args()


def resolve_data_path(data_root: str, path: str) -> Path:
    return Path(osp.join(data_root, osp.relpath(path, "./data")))


def resolve_eval_files(eval_root: str):
    revos_dir = Path(eval_root) / "refvos" / "REVOS"
    pred_path = revos_dir / "results.json"
    csv_path = revos_dir / "revos_valid.csv"
    return pred_path, csv_path


def decode_rle_mask(rle_obj):
    mask = cocomask.decode(rle_obj)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8)


def load_csv_rows(csv_path: Path):
    with csv_path.open() as f:
        return {row["videxp"]: row for row in csv.DictReader(f)}


def parse_case_string(case_str: str):
    if ":" not in case_str:
        raise ValueError(f"Invalid case format: {case_str}. Expected video_id:exp_id")
    video_id, exp_id = case_str.rsplit(":", 1)
    return video_id, exp_id


def build_case_list(case_args):
    case_pairs = [parse_case_string(case_str) for case_str in case_args]
    deduped = []
    seen = set()
    for item in case_pairs:
        if item in seen:
            continue
        deduped.append(item)
        seen.add(item)
    return deduped


def export_case(
    video_id,
    exp_id,
    meta,
    mask_dict,
    ours_pred,
    baseline_pred,
    ours_rows,
    baseline_rows,
    image_root,
    output_dir,
):
    video = meta[video_id]
    exp = video["expressions"][exp_id]
    frames = video["frames"]
    anno_ids = exp["anno_id"]

    ours_item = ours_pred[video_id][exp_id]
    baseline_item = baseline_pred[video_id][exp_id]
    ours_masks = [decode_rle_mask(rle) for rle in ours_item["prediction_masks"]]
    baseline_masks = [decode_rle_mask(rle) for rle in baseline_item["prediction_masks"]]

    case_dir = output_dir / video_id.replace("/", "__") / f"exp_{exp_id}"
    frames_dir = case_dir / "frames"
    gt_dir = case_dir / "gt_masks"
    ours_dir = case_dir / "ours_masks"
    baseline_dir = case_dir / "baseline_masks"
    for path in [frames_dir, gt_dir, ours_dir, baseline_dir]:
        path.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_name in enumerate(frames):
        frame_path = image_root / video_id / f"{frame_name}.jpg"
        if not frame_path.exists():
            frame_path = image_root / video_id / f"{frame_name}.png"
        if not frame_path.exists():
            raise FileNotFoundError(f"Frame not found: {frame_path}")

        image = Image.open(frame_path).convert("RGB")
        image.save(frames_dir / f"{frame_name}.png")

        gt_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        for anno_id in anno_ids:
            mask_rle = mask_dict[str(anno_id)][frame_idx]
            if not mask_rle:
                continue
            decoded = cocomask.decode(mask_rle)
            if decoded.ndim == 3:
                decoded = decoded.sum(axis=2)
            gt_mask = np.maximum(gt_mask, (decoded > 0).astype(np.uint8))

        Image.fromarray(gt_mask * 255).save(gt_dir / f"{frame_name}.png")
        Image.fromarray(ours_masks[frame_idx] * 255).save(ours_dir / f"{frame_name}.png")
        Image.fromarray(baseline_masks[frame_idx] * 255).save(
            baseline_dir / f"{frame_name}.png"
        )

    videxp = f"{video_id}_{exp_id}"
    ours_row = ours_rows.get(videxp)
    baseline_row = baseline_rows.get(videxp)
    meta_obj = {
        "video_id": video_id,
        "exp_id": exp_id,
        "prompt": exp["exp"],
        "type_id": exp["type_id"],
        "anno_id": anno_ids,
        "num_frames": len(frames),
        "frames": frames,
    }
    if ours_row is not None and baseline_row is not None:
        meta_obj["scores"] = {
            "ours": {k: ours_row[k] for k in ["J", "F", "JF", "A", "R"]},
            "baseline": {k: baseline_row[k] for k in ["J", "F", "JF", "A", "R"]},
            "delta": {
                "J": round(float(ours_row["J"]) - float(baseline_row["J"]), 2),
                "F": round(float(ours_row["F"]) - float(baseline_row["F"]), 2),
                "JF": round(float(ours_row["JF"]) - float(baseline_row["JF"]), 2),
            },
        }

    with (case_dir / "meta.json").open("w") as f:
        json.dump(meta_obj, f, indent=2, ensure_ascii=False)
    return meta_obj


def save_summary(summary, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (output_dir / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video_id",
                "exp_id",
                "prompt",
                "type_id",
                "num_frames",
                "ours_JF",
                "base_JF",
                "delta_JF",
                "ours_J",
                "base_J",
                "delta_J",
                "ours_F",
                "base_F",
                "delta_F",
            ]
        )
        for item in summary:
            scores = item.get("scores", {})
            ours = scores.get("ours", {})
            baseline = scores.get("baseline", {})
            delta = scores.get("delta", {})
            writer.writerow(
                [
                    item["video_id"],
                    item["exp_id"],
                    item["prompt"],
                    item["type_id"],
                    item["num_frames"],
                    ours.get("JF", ""),
                    baseline.get("JF", ""),
                    delta.get("JF", ""),
                    ours.get("J", ""),
                    baseline.get("J", ""),
                    delta.get("J", ""),
                    ours.get("F", ""),
                    baseline.get("F", ""),
                    delta.get("F", ""),
                ]
            )


def main():
    args = parse_args()
    exp_path = resolve_data_path(args.data_root, args.exp_path)
    mask_path = resolve_data_path(args.data_root, args.mask_path)
    image_root = resolve_data_path(args.data_root, args.image_root)
    output_dir = Path(args.output_dir) / "visualize" / "revos_case_pack_manual"
    ours_pred_path, ours_csv_path = resolve_eval_files(args.ours_eval_root)
    baseline_pred_path, baseline_csv_path = resolve_eval_files(args.baseline_eval_root)

    with exp_path.open() as f:
        meta = json.load(f)["videos"]
    with mask_path.open() as f:
        mask_dict = json.load(f)
    with ours_pred_path.open() as f:
        ours_pred = json.load(f)
    with baseline_pred_path.open() as f:
        baseline_pred = json.load(f)

    ours_rows = load_csv_rows(ours_csv_path)
    baseline_rows = load_csv_rows(baseline_csv_path)

    case_pairs = build_case_list(args.case)
    if not case_pairs:
        raise ValueError("No cases selected. Use --case video_id:exp_id at least once.")

    summary = []
    for video_id, exp_id in case_pairs:
        summary.append(
            export_case(
                video_id=video_id,
                exp_id=exp_id,
                meta=meta,
                mask_dict=mask_dict,
                ours_pred=ours_pred,
                baseline_pred=baseline_pred,
                ours_rows=ours_rows,
                baseline_rows=baseline_rows,
                image_root=image_root,
                output_dir=output_dir,
            )
        )

    save_summary(summary, output_dir)

    print(f"Exported {len(summary)} cases to {output_dir}")
    for item in summary:
        score = item.get("scores", {}).get("delta", {}).get("JF", "NA")
        print(f"{item['video_id']}:{item['exp_id']} delta_JF={score}")


if __name__ == "__main__":
    main()
