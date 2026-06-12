###########################################################################
# Created by: OpenAI Codex
# Based on the original MeViS evaluation script.
###########################################################################
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)
import os.path as osp
import time
import argparse
import json
import csv
import numpy as np
from pycocotools import mask as cocomask
from third_parts.revos.utils.metircs import db_eval_iou, db_eval_boundary
import multiprocessing as mp

NUM_WOEKERS = 128


def eval_queue(q, rank, out_dict):
    while not q.empty():
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]
        exp_name = f"{vid_name}_{exp}"
        pred = mask_pred_dict[vid_name][exp]

        h, w = pred["prediction_masks"][0]["size"]
        vid_len = len(vid["frames"])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        anno_ids = vid["expressions"][exp]["anno_id"]

        for frame_idx, frame_name in enumerate(vid["frames"]):
            for anno_id in anno_ids:
                mask_rle = mask_dict[str(anno_id)][frame_idx]
                if mask_rle:
                    gt_masks[frame_idx] += cocomask.decode(mask_rle)

            pred_mask = cocomask.decode(pred["prediction_masks"][frame_idx])
            pred_masks[frame_idx] += pred_mask

        j = db_eval_iou(gt_masks, pred_masks).mean()
        f = db_eval_boundary(gt_masks, pred_masks).mean()
        out_dict[exp_name] = [j, f]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_path", type=str)
    parser.add_argument(
        "--mevis_exp_path",
        type=str,
        default="./data/video_datas/mevis/valid_u/meta_expressions.json",
    )
    parser.add_argument(
        "--mevis_mask_path",
        type=str,
        default="./data/video_datas/mevis/valid_u/mask_dict.json",
    )
    parser.add_argument("--data_root", default="./data", help="Root directory for all datasets.")
    parser.add_argument("--save_name", type=str, default="mevis_valu.json")
    parser.add_argument("--save_csv_name", type=str, default="mevis_valu.csv")
    args = parser.parse_args()
    if args.data_root:
        args.mevis_exp_path = osp.join(args.data_root, osp.relpath(args.mevis_exp_path, "./data"))
        args.mevis_mask_path = osp.join(args.data_root, osp.relpath(args.mevis_mask_path, "./data"))

    queue = mp.Queue()
    exp_dict = json.load(open(args.mevis_exp_path))["videos"]
    mask_dict = json.load(open(args.mevis_mask_path))

    shared_exp_dict = mp.Manager().dict(exp_dict)
    shared_mask_dict = mp.Manager().dict(mask_dict)
    output_dict = mp.Manager().dict()

    mask_pred = json.load(open(args.pred_path))
    mask_pred_dict = mp.Manager().dict(mask_pred)

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid["expressions"]:
            queue.put([vid_name, exp])

    start_time = time.time()
    if NUM_WOEKERS > 1:
        processes = []
        for rank in range(NUM_WOEKERS):
            p = mp.Process(target=eval_queue, args=(queue, rank, output_dict))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        eval_queue(queue, 0, output_dict)

    data_list = []
    for videxp, (j, f) in output_dict.items():
        vid_name, exp_id = videxp.rsplit("_", 1)
        data_list.append(
            {
                "video_name": vid_name,
                "exp_id": exp_id,
                "exp": exp_dict[vid_name]["expressions"][exp_id]["exp"],
                "videxp": videxp,
                "J": round(100 * float(j), 2),
                "F": round(100 * float(f), 2),
                "JF": round(100 * float((j + f) / 2), 2),
            }
        )

    j = [item["J"] for item in data_list]
    f = [item["F"] for item in data_list]

    output_path = osp.join(osp.dirname(args.pred_path), args.save_name)
    output_csv_path = osp.join(osp.dirname(args.pred_path), args.save_csv_name)
    results = {
        "J": round(float(np.mean(j)), 2),
        "F": round(float(np.mean(f)), 2),
        "J&F": round(float((np.mean(j) + np.mean(f)) / 2), 2),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(output_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_name", "exp_id", "exp", "videxp", "J", "F", "JF"],
        )
        writer.writeheader()
        writer.writerows(data_list)

    print(json.dumps(results, indent=4))
    print(f"Results saved to {output_path}")
    print(f"Results saved to {output_csv_path}")

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
