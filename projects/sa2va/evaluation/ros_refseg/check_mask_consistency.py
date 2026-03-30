import argparse
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from pycocotools import mask as mask_utils

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


def decode_mask_training_style(seg, height: int, width: int) -> np.ndarray:
    if isinstance(seg, dict):
        decoded = mask_utils.decode(seg)
    elif isinstance(seg, str):
        decoded = mask_utils.decode({'size': [height, width], 'counts': seg})
    elif isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((height, width), dtype=np.uint8)
        if isinstance(seg[0], (int, float)):
            rles = mask_utils.frPyObjects([seg], height, width)
        else:
            rles = mask_utils.frPyObjects(seg, height, width)
        decoded = mask_utils.decode(rles)
    else:
        raise TypeError(f'Unsupported segmentation type: {type(seg)}')

    decoded = decoded.astype(np.uint8)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2).astype(np.uint8)
    return decoded


def align_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    h, w = mask.shape
    if (h, w) == (height, width):
        return mask

    out = np.zeros((height, width), dtype=np.uint8)
    copy_h = min(h, height)
    copy_w = min(w, width)
    out[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
    return out


def build_mask_training_style(ref, refer: REFER) -> Optional[np.ndarray]:
    image = refer.Imgs[ref['image_id']]
    height, width = int(image['height']), int(image['width'])
    ann = refer.refToAnn.get(ref['ref_id'], None)
    if ann is None:
        return None

    seg = ann.get('segmentation', None)
    if seg is None:
        return None

    try:
        binary_mask = np.zeros((height, width), dtype=np.uint8)

        if isinstance(seg, dict):
            seg_iter = [seg]
        elif isinstance(seg, str):
            seg_iter = [seg]
        elif isinstance(seg, list):
            if len(seg) == 0:
                return None
            if isinstance(seg[0], (int, float)):
                seg_iter = [seg]
            elif isinstance(seg[0], dict):
                seg_iter = seg
            else:
                seg_iter = seg
        else:
            return None

        for seg_item in seg_iter:
            decoded = decode_mask_training_style(seg_item, height, width)
            decoded = align_mask(decoded, height, width)
            binary_mask = np.clip(binary_mask + decoded, 0, 1)

        if binary_mask.sum() == 0:
            return None
        return binary_mask.astype(np.uint8)
    except Exception:
        return None


def build_mask_refer_legacy_style(ref, refer: REFER) -> Optional[np.ndarray]:
    try:
        m = refer.getMask(ref)['mask']
    except Exception:
        return None
    if m is None:
        return None
    m = np.asarray(m)
    if m.ndim == 3:
        m = m.any(axis=2).astype(np.uint8)
    else:
        m = (m > 0).astype(np.uint8)
    if m.sum() == 0:
        return None
    return m


def iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0).astype(np.uint8)
    bb = (b > 0).astype(np.uint8)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    return float(inter) / float(union) if union > 0 else 1.0


def parse_args():
    parser = argparse.ArgumentParser(description='Check GT mask consistency between training-style and legacy refer parser')
    parser.add_argument('--dataset', choices=['RRSISD', 'ris_lad'], required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--data_root', type=str, default='./data/ROS-Sa2VA')
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_root', type=str, default='./work_dirs/ros_refseg_eval')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--save_diff_cases', type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    dataset_cfg = DATASET_MAP[args.dataset]
    refer_data_root = os.path.join(args.data_root, dataset_cfg['refer_subdir'])
    refer = REFER(refer_data_root, dataset=dataset_cfg['dataset_name'], splitBy='unc')

    ref_ids = refer.getRefIds(split=args.split)
    if len(ref_ids) == 0:
        raise RuntimeError(f'No refs for dataset={args.dataset} split={args.split}')

    total_refs = len(ref_ids)
    sample_n = min(args.sample_size, total_refs)
    sampled_ref_ids = random.sample(ref_ids, sample_n)

    ts = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.run_name or f'consistency_{args.dataset}_{args.split}_{sample_n}_{ts}'
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    rows: List[Dict] = []
    perfect = 0
    both_none = 0
    one_none = 0

    for ref_id in sampled_ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        m_train = build_mask_training_style(ref, refer)
        m_legacy = build_mask_refer_legacy_style(ref, refer)

        if m_train is None and m_legacy is None:
            both_none += 1
            rows.append({'ref_id': int(ref_id), 'status': 'both_none', 'iou': None})
            continue

        if m_train is None or m_legacy is None:
            one_none += 1
            rows.append({'ref_id': int(ref_id), 'status': 'one_none', 'iou': 0.0})
            continue

        # Align shape if metadata/path irregularities exist.
        if m_train.shape != m_legacy.shape:
            h = max(m_train.shape[0], m_legacy.shape[0])
            w = max(m_train.shape[1], m_legacy.shape[1])
            m_train = align_mask(m_train, h, w)
            m_legacy = align_mask(m_legacy, h, w)

        cur_iou = iou(m_train, m_legacy)
        if abs(cur_iou - 1.0) < 1e-9:
            perfect += 1

        rows.append(
            {
                'ref_id': int(ref_id),
                'status': 'ok',
                'iou': round(cur_iou, 8),
                'train_area': int((m_train > 0).sum()),
                'legacy_area': int((m_legacy > 0).sum()),
                'image_id': int(ref['image_id']),
            }
        )

    valid_ious = [x['iou'] for x in rows if x['status'] == 'ok' and x['iou'] is not None]
    mean_iou = float(np.mean(valid_ious)) if len(valid_ious) > 0 else 0.0

    summary = {
        'dataset': args.dataset,
        'split': args.split,
        'total_refs_split': total_refs,
        'sample_size': sample_n,
        'seed': args.seed,
        'perfect_match_count': perfect,
        'perfect_match_rate': round(100.0 * perfect / sample_n, 2) if sample_n > 0 else 0.0,
        'both_none_count': both_none,
        'one_none_count': one_none,
        'mean_iou_on_valid': round(mean_iou, 6),
    }

    diff_rows = [x for x in rows if not (x['status'] == 'ok' and abs(float(x['iou']) - 1.0) < 1e-9)]
    diff_rows = diff_rows[: max(0, args.save_diff_cases)]

    summary_path = os.path.join(run_dir, 'summary.json')
    details_path = os.path.join(run_dir, 'details.jsonl')
    diff_path = os.path.join(run_dir, 'diff_cases.json')

    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(details_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    with open(diff_path, 'w', encoding='utf-8') as f:
        json.dump(diff_rows, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Saved: {summary_path}')
    print(f'Saved: {details_path}')
    print(f'Saved: {diff_path}')


if __name__ == '__main__':
    main()
