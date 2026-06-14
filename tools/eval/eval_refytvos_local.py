###########################################################################
# Created by: OpenAI Codex
# Local Ref-YTVOS evaluator for the released valid_local subset.
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import argparse
import csv
import json
import multiprocessing as mp
import os.path as osp
import time
from pathlib import Path

import numpy as np
from PIL import Image

from third_parts.revos.utils.metircs import db_eval_boundary, db_eval_iou

NUM_WOEKERS = 128
exp_dict = None
pred_root = None
gt_root = None


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path), dtype=np.uint8)
    return (mask > 0).astype(np.uint8)


def get_prediction_root(pred_path):
    pred_path = Path(pred_path)
    if pred_path.is_dir() and pred_path.name == "Annotations":
        return pred_path, pred_path.parent
    if pred_path.is_dir() and (pred_path / "Annotations").is_dir():
        return pred_path / "Annotations", pred_path
    raise FileNotFoundError(
        f"Cannot find prediction Annotations directory from: {pred_path}"
    )


def eval_queue(q, rank, out_dict):
    while not q.empty():
        vid_name, exp_id = q.get()
        vid = exp_dict[vid_name]
        exp_name = f"{vid_name}_{exp_id}"

        pred_dir = pred_root / vid_name / exp_id
        gt_dir = gt_root / vid_name / exp_id
        if not pred_dir.is_dir():
            raise FileNotFoundError(f"Missing prediction directory: {pred_dir}")
        if not gt_dir.is_dir():
            raise FileNotFoundError(f"Missing GT directory: {gt_dir}")

        frame_names = vid["frames"]
        first_gt = gt_dir / f"{frame_names[0]}.png"
        if not first_gt.is_file():
            raise FileNotFoundError(f"Missing GT mask: {first_gt}")

        h, w = load_mask(first_gt).shape
        vid_len = len(frame_names)
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        for frame_idx, frame_name in enumerate(frame_names):
            gt_mask_path = gt_dir / f"{frame_name}.png"
            pred_mask_path = pred_dir / f"{frame_name}.png"
            if not gt_mask_path.is_file():
                raise FileNotFoundError(f"Missing GT mask: {gt_mask_path}")
            if not pred_mask_path.is_file():
                raise FileNotFoundError(f"Missing prediction mask: {pred_mask_path}")

            gt_masks[frame_idx] = load_mask(gt_mask_path)
            pred_masks[frame_idx] = load_mask(pred_mask_path)

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "pred_path",
        type=str,
        help="Path to prediction Annotations dir or its parent work dir.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./data/video_datas/rvos/valid_local/meta_expressions_challenge.json",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="./data/video_datas/rvos/valid_local/Annotations",
    )
    parser.add_argument(
        "--data_root", default="./data", help="Root directory for all datasets."
    )
    parser.add_argument(
        "--save_json_name", type=str, default="refytvos_valid_local.json"
    )
    parser.add_argument(
        "--save_csv_name", type=str, default="refytvos_valid_local.csv"
    )
    args = parser.parse_args()

    if args.data_root:
        args.exp_path = osp.join(args.data_root, osp.relpath(args.exp_path, "./data"))
        args.gt_path = osp.join(args.data_root, osp.relpath(args.gt_path, "./data"))

    queue = mp.Queue()
    exp_dict = json.load(open(args.exp_path))["videos"]
    pred_root, output_root = get_prediction_root(args.pred_path)
    gt_root = Path(args.gt_path)
    output_dict = mp.Manager().dict()

    for vid_name, vid in exp_dict.items():
        for exp_id in vid["expressions"]:
            queue.put([vid_name, exp_id])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Worker {p.pid} exited with code {p.exitcode}")
    else:
        eval_queue(queue, 0, output_dict)

    data_list = []
    for videxp, (j, f) in output_dict.items():
        vid_name, exp_id = videxp.rsplit("_", 1)
        expr = exp_dict[vid_name]["expressions"][exp_id]
        data_list.append(
            {
                "video_name": vid_name,
                "exp_id": exp_id,
                "obj_id": expr.get("obj_id"),
                "exp": expr["exp"],
                "videxp": videxp,
                "J": round(100 * float(j), 2),
                "F": round(100 * float(f), 2),
                "JF": round(100 * float((j + f) / 2), 2),
            }
        )

    data_list.sort(key=lambda x: (x["video_name"], int(x["exp_id"])))
    j = [item["J"] for item in data_list]
    f = [item["F"] for item in data_list]

    output_json_path = output_root / args.save_json_name
    output_csv_path = output_root / args.save_csv_name
    results = {
        "num_videos": len(exp_dict),
        "num_expressions": len(data_list),
        "J": round(float(np.mean(j)), 2),
        "F": round(float(np.mean(f)), 2),
        "J&F": round(float((np.mean(j) + np.mean(f)) / 2), 2),
    }

    with open(output_json_path, "w") as fobj:
        json.dump(results, fobj, indent=4)

    with open(output_csv_path, "w", newline="") as fobj:
        writer = csv.DictWriter(
            fobj,
            fieldnames=["video_name", "exp_id", "obj_id", "exp", "videxp", "J", "F", "JF"],
        )
        writer.writeheader()
        writer.writerows(data_list)

    print(json.dumps(results, indent=4))
    print(f"Results saved to {output_json_path}")
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % total_time)
