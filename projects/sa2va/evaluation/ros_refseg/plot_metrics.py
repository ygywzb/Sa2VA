import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

PR_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load_jsonl(path):
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description='Plot ROS-RefSeg metrics from saved files')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--dataset', choices=['RRSISD', 'ris_lad'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--out_subdir', type=str, default='plots')
    return parser.parse_args()


def main():
    args = parse_args()

    metrics_path = os.path.join(args.run_dir, f'metrics_{args.dataset}_{args.split}.json')
    details_path = os.path.join(args.run_dir, f'details_{args.dataset}_{args.split}.jsonl')

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f'Missing metrics: {metrics_path}')
    if not os.path.exists(details_path):
        raise FileNotFoundError(f'Missing details: {details_path}')

    with open(metrics_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)
    rows = load_jsonl(details_path)

    ious = np.array([float(x['iou']) for x in rows], dtype=np.float64)

    out_dir = os.path.join(args.run_dir, args.out_subdir, f'{args.dataset}_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    # 1) IoU histogram
    plt.figure(figsize=(7, 4.5), dpi=140)
    plt.hist(ious, bins=30, color='#2a9d8f', edgecolor='white')
    plt.title(f'IoU Distribution ({args.dataset} {args.split})')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.tight_layout()
    hist_path = os.path.join(out_dir, 'iou_hist.png')
    plt.savefig(hist_path)
    plt.close()

    # 2) PR@threshold curve in percentage
    if 'percent' in metrics:
        pr_keys = [k for k in metrics['percent'].keys() if k.startswith('Pr@')]
        pr_keys = sorted(pr_keys, key=lambda x: float(x.split('@')[1]))
        x = [float(k.split('@')[1]) for k in pr_keys]
        y = [float(metrics['percent'][k]) for k in pr_keys]
    else:
        x = PR_THRESHOLDS
        y = [round(float(np.mean(ious >= t)) * 100.0, 2) for t in x]

    plt.figure(figsize=(7, 4.5), dpi=140)
    plt.plot(x, y, marker='o', linewidth=2, color='#e76f51')
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.5, f'{yi:.2f}', ha='center', fontsize=8)
    plt.title(f'Pr@k Curve ({args.dataset} {args.split})')
    plt.xlabel('IoU threshold k')
    plt.ylabel('Precision (%)')
    plt.ylim(0, 100)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    pr_path = os.path.join(out_dir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()

    # 3) Cumulative mIoU curve
    ord_idx = np.arange(1, len(ious) + 1)
    cum_miou = np.cumsum(ious) / ord_idx
    plt.figure(figsize=(7, 4.5), dpi=140)
    plt.plot(ord_idx, cum_miou * 100.0, color='#457b9d', linewidth=2)
    plt.title(f'Cumulative mIoU ({args.dataset} {args.split})')
    plt.xlabel('Samples')
    plt.ylabel('mIoU (%)')
    plt.grid(alpha=0.25)
    plt.tight_layout()
    cm_path = os.path.join(out_dir, 'cum_miou.png')
    plt.savefig(cm_path)
    plt.close()

    print(f'Saved: {hist_path}')
    print(f'Saved: {pr_path}')
    print(f'Saved: {cm_path}')


if __name__ == '__main__':
    main()
