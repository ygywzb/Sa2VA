import argparse
import glob
import json
import os
import textwrap
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from projects.sa2va.ReSaP.utils.RRSISD_dataset import REFER


DATASET_MAP = {
    'RRSISD': {
        'refer_subdir': 'RRSIS-D',
        'dataset_name': 'rrsisd',
    },
    'ris_lad': {
        'refer_subdir': 'RIS-LAD',
        'dataset_name': 'ris_lad',
    },
}


def to_hw_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[0]
    return (mask > 0).astype(np.uint8)


def overlay_color(image_np, mask, color=(255, 0, 0), alpha=0.45):
    out = image_np.copy().astype(np.float32)
    m = mask.astype(bool)
    c = np.array(color, dtype=np.float32)
    out[m] = (1 - alpha) * out[m] + alpha * c
    return out.astype(np.uint8)


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def locate_pred_mask(run_dir: str, item: Dict):
    if item.get('pred_mask_path') and os.path.exists(item['pred_mask_path']):
        return item['pred_mask_path']

    dataset = item.get('dataset', '')
    split = item.get('split', '')
    pred_dir = os.path.join(run_dir, f'pred_masks_{dataset}_{split}')
    if not os.path.isdir(pred_dir):
        # Backward compatibility for previous runs saved by ReSaP eval.py.
        pred_dir = os.path.join(run_dir, 'pred_masks')
    idx = int(item['index'])
    ref_id = int(item['ref_id'])

    exact = os.path.join(pred_dir, f'0_{idx}_{ref_id}.png')
    if os.path.exists(exact):
        return exact

    cands = glob.glob(os.path.join(pred_dir, f'*_{idx}_{ref_id}.png'))
    return cands[0] if len(cands) > 0 else None


def select_cases(rows: List[Dict], mode: str, num_cases: int) -> List[Dict]:
    rows_sorted = sorted(rows, key=lambda x: float(x['iou']))
    n = len(rows_sorted)

    if n == 0:
        return []

    if mode == 'high':
        return rows_sorted[-num_cases:][::-1]
    if mode == 'low':
        return rows_sorted[:num_cases]
    if mode == 'mid':
        center = n // 2
        half = num_cases // 2
        start = max(0, center - half)
        out = rows_sorted[start:start + num_cases]
        if len(out) < num_cases:
            out = rows_sorted[max(0, n - num_cases):n]
        return out

    # mixed
    low_n = num_cases // 2
    high_n = num_cases - low_n
    return rows_sorted[-high_n:][::-1] + rows_sorted[:low_n]


def draw_case_figure(item: Dict, refer: REFER, run_dir: str, out_path: str):
    ref_id = int(item['ref_id'])
    idx = int(item['index'])

    img = Image.open(item['image_path']).convert('RGB')
    img_np = np.array(img)

    ref = refer.loadRefs(ref_id)[0]
    gt_mask = to_hw_mask(refer.getMask(ref)['mask'])

    pred_path = locate_pred_mask(run_dir, item)
    if pred_path is None:
        pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)
    else:
        pred_mask = to_hw_mask(np.array(Image.open(pred_path)))

    if pred_mask.shape != gt_mask.shape:
        pred_mask = np.array(Image.fromarray(pred_mask).resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST))
        pred_mask = to_hw_mask(pred_mask)

    tp = (pred_mask & gt_mask).astype(np.uint8)
    fp = (pred_mask & (1 - gt_mask)).astype(np.uint8)
    fn = (gt_mask & (1 - pred_mask)).astype(np.uint8)

    pred_overlay = overlay_color(img_np, pred_mask, color=(255, 80, 80), alpha=0.48)
    gt_overlay = overlay_color(img_np, gt_mask, color=(80, 255, 80), alpha=0.48)

    err = img_np.copy().astype(np.float32)
    err[tp.astype(bool)] = 0.55 * err[tp.astype(bool)] + 0.45 * np.array([70, 220, 70], dtype=np.float32)
    err[fp.astype(bool)] = 0.55 * err[fp.astype(bool)] + 0.45 * np.array([240, 60, 60], dtype=np.float32)
    err[fn.astype(bool)] = 0.55 * err[fn.astype(bool)] + 0.45 * np.array([70, 70, 245], dtype=np.float32)
    err = err.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(19, 5), dpi=140)
    axes[0].imshow(img_np)
    axes[0].set_title('Image')
    axes[1].imshow(pred_overlay)
    axes[1].set_title('Prediction (red)')
    axes[2].imshow(gt_overlay)
    axes[2].set_title('Ground truth (green)')
    axes[3].imshow(err)
    axes[3].set_title('TP=green FP=red FN=blue')

    for ax in axes:
        ax.axis('off')

    target_text = textwrap.shorten(item.get('text', ''), width=120, placeholder='...')
    pred_text = textwrap.shorten(str(item.get('prediction', '')).replace('\n', ' '), width=130, placeholder='...')

    header = (
        f"IoU={float(item['iou']):.4f} | inter={int(item['inter'])} union={int(item['union'])} "
        f"| index={idx} ref_id={ref_id} image_id={int(item['image_id'])}"
    )
    footer1 = f'Target object: {target_text}'
    footer2 = f'Prediction text: {pred_text}'

    fig.suptitle(header, fontsize=10, y=0.98)
    fig.text(0.01, 0.04, footer1, fontsize=9)
    fig.text(0.01, 0.015, footer2, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(out_path)
    plt.close(fig)

    return {
        'index': idx,
        'ref_id': ref_id,
        'image_id': int(item['image_id']),
        'iou': float(item['iou']),
        'inter': int(item['inter']),
        'union': int(item['union']),
        'text': item.get('text', ''),
        'prediction': item.get('prediction', ''),
        'figure': out_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize top/low/mid/mixed ROS-RefSeg cases from saved details')
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--dataset', choices=['RRSISD', 'ris_lad'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], required=True)
    parser.add_argument('--data_root', type=str, default='./data/ROS-Sa2VA')
    parser.add_argument('--mode', choices=['high', 'low', 'mid', 'mixed'], default='mixed')
    parser.add_argument('--num_cases', type=int, default=4)
    parser.add_argument('--min_iou', type=float, default=None)
    parser.add_argument('--max_iou', type=float, default=None)
    parser.add_argument('--exclude_zero_iou', action='store_true')
    parser.add_argument('--out_subdir', type=str, default='visualizations')
    return parser.parse_args()


def main():
    args = parse_args()
    details_path = os.path.join(args.run_dir, f'details_{args.dataset}_{args.split}.jsonl')
    if not os.path.exists(details_path):
        raise FileNotFoundError(f'Missing details file: {details_path}')

    rows = load_jsonl(details_path)

    if args.exclude_zero_iou:
        rows = [x for x in rows if float(x['iou']) > 0.0]
    if args.min_iou is not None:
        rows = [x for x in rows if float(x['iou']) >= args.min_iou]
    if args.max_iou is not None:
        rows = [x for x in rows if float(x['iou']) <= args.max_iou]

    if len(rows) == 0:
        raise RuntimeError('No samples left after IoU filtering. Please relax filter options.')

    selected = select_cases(rows, args.mode, args.num_cases)

    dataset_cfg = DATASET_MAP[args.dataset]
    refer_data_root = os.path.join(args.data_root, dataset_cfg['refer_subdir'])
    refer = REFER(refer_data_root, dataset=dataset_cfg['dataset_name'], splitBy='unc')

    out_dir = os.path.join(args.run_dir, args.out_subdir, f'{args.dataset}_{args.split}_{args.mode}_{args.num_cases}')
    os.makedirs(out_dir, exist_ok=True)

    report = {
        'dataset': args.dataset,
        'split': args.split,
        'mode': args.mode,
        'num_cases': args.num_cases,
        'filters': {
            'exclude_zero_iou': args.exclude_zero_iou,
            'min_iou': args.min_iou,
            'max_iou': args.max_iou,
        },
        'num_candidates_after_filter': len(rows),
        'cases': [],
    }

    for i, item in enumerate(selected, start=1):
        fig_path = os.path.join(out_dir, f'case_{i:02d}_iou_{float(item["iou"]):.4f}.png')
        report['cases'].append(draw_case_figure(item, refer, args.run_dir, fig_path))

    # Build a compact grid overview
    n = len(report['cases'])
    cols = 2
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 5 * rows_n), dpi=140)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for ax, item in zip(axes, report['cases']):
        img = Image.open(item['figure']).convert('RGB')
        ax.imshow(np.array(img))
        title_text = textwrap.shorten(item.get('text', ''), width=48, placeholder='...')
        ax.set_title(f"IoU={item['iou']:.4f} | {title_text}", fontsize=9)
        ax.axis('off')

    for ax in axes[n:]:
        ax.axis('off')

    overview_path = os.path.join(out_dir, 'overview.png')
    plt.tight_layout()
    plt.savefig(overview_path)
    plt.close(fig)

    report_path = os.path.join(out_dir, 'selected_cases.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f'Saved: {report_path}')
    print(f'Saved: {overview_path}')


if __name__ == '__main__':
    main()
