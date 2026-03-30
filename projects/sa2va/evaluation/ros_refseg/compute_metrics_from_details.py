import argparse
import json
import os
from typing import Dict, List

import numpy as np

PR_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def load_details(details_path: str) -> List[Dict]:
    rows = []
    with open(details_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: List[Dict], thresholds: List[float]) -> Dict:
    ious = np.array([float(x['iou']) for x in rows], dtype=np.float64) if len(rows) > 0 else np.array([], dtype=np.float64)
    total_inter = int(np.sum([int(x['inter']) for x in rows])) if len(rows) > 0 else 0
    total_union = int(np.sum([int(x['union']) for x in rows])) if len(rows) > 0 else 0

    m_iou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    o_iou = float(total_inter) / float(total_union) if total_union > 0 else 0.0

    ratio = {'mIoU': m_iou, 'oIoU': o_iou}
    percent = {'mIoU': round(100.0 * m_iou, 2), 'oIoU': round(100.0 * o_iou, 2)}

    for t in thresholds:
        key = f'Pr@{t:.1f}'
        val = float(np.mean(ious >= t)) if len(ious) > 0 else 0.0
        ratio[key] = val
        percent[key] = round(100.0 * val, 2)

    return {
        'num_samples': len(rows),
        'thresholds': thresholds,
        'ratio': ratio,
        'percent': percent,
        'total_inter': total_inter,
        'total_union': total_union,
        'format_note': 'percent metrics are xx.xx values (0-100 scale)',
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Compute ROS-RefSeg metrics from details jsonl only')
    parser.add_argument('--details_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='')
    return parser.parse_args()


def main():
    args = parse_args()
    rows = load_details(args.details_path)
    metrics = compute_metrics(rows, PR_THRESHOLDS)

    output_path = args.output_path
    if output_path == '':
        base = os.path.splitext(args.details_path)[0]
        output_path = f'{base}_recomputed_metrics.json'

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics['percent'], ensure_ascii=False, indent=2))
    print(f'Saved: {output_path}')


if __name__ == '__main__':
    main()
