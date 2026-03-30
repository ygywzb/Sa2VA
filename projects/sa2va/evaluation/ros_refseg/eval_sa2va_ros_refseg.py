import argparse
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from pycocotools import mask as mask_utils
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from projects.sa2va.ReSaP.utils.RRSISD_dataset import REFER
from projects.sa2va.evaluation.utils import _init_dist_pytorch, collect_results_cpu, get_dist_info, get_rank


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

PR_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def find_seg_indices(text: str) -> Tuple[List[int], List[int]]:
    all_seg_indices = [m.start() for m in re.finditer(r'\[SEG\]', text)]
    answer_spans = [(m.start(), m.end()) for m in re.finditer(r'<answer>.*?</answer>', text, re.DOTALL)]
    if len(answer_spans) == 0:
        return [], []
    start, end = answer_spans[0]

    seg_indices_in_reason, seg_indices_in_answer = [], []
    for idx, seg_ind in enumerate(all_seg_indices):
        if start <= seg_ind < end:
            seg_indices_in_answer.append(idx)
        elif seg_ind < start:
            seg_indices_in_reason.append(idx)
    return seg_indices_in_reason, seg_indices_in_answer


class ROSRefSegEvalDataset(Dataset):
    def __init__(self, data_root: str, option: str, split: str = 'test', with_thinking: bool = False):
        dataset_cfg = DATASET_MAP[option]
        refer_data_root = os.path.join(data_root, dataset_cfg['refer_subdir'])
        self.refer = REFER(refer_data_root, dataset=dataset_cfg['dataset_name'], splitBy='unc')
        self.with_thinking = with_thinking
        self.samples = self._build_samples(split)

    def _build_samples(self, split: str):
        samples = []
        ref_ids = self.refer.getRefIds(split=split)
        skipped_invalid_mask = 0
        skipped_missing_image = 0
        for ref_id in ref_ids:
            ref = self.refer.loadRefs(ref_id)[0]
            image_info = self.refer.Imgs[ref['image_id']]
            image_path = self._resolve_image_path(image_info['file_name'])
            if not os.path.exists(image_path):
                skipped_missing_image += len(ref['sentences'])
                continue

            gt_mask = self._build_gt_mask_from_ref(ref)
            if gt_mask is None:
                skipped_invalid_mask += len(ref['sentences'])
                continue

            for sent in ref['sentences']:
                text = sent['sent']
                samples.append(
                    {
                        'ref_id': ref_id,
                        'image_id': ref['image_id'],
                        'image_path': image_path,
                        'text': self._format_prompt(text),
                        'raw_text': text,
                        'gt_mask': gt_mask,
                    }
                )

        if skipped_missing_image > 0 or skipped_invalid_mask > 0:
            print(
                f'[ROSRefSegEvalDataset] split={split} kept={len(samples)} '
                f'skipped_missing_image={skipped_missing_image} '
                f'skipped_invalid_mask={skipped_invalid_mask}'
            )
        return samples

    def _decode_mask(self, seg, height: int, width: int) -> np.ndarray:
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

    def _align_mask_to_image_size(self, mask: np.ndarray, height: int, width: int) -> np.ndarray:
        h, w = mask.shape
        if (h, w) == (height, width):
            return mask

        aligned = np.zeros((height, width), dtype=np.uint8)
        copy_h = min(h, height)
        copy_w = min(w, width)
        aligned[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
        return aligned

    def _build_gt_mask_from_ref(self, ref) -> Optional[np.ndarray]:
        image = self.refer.Imgs[ref['image_id']]
        height, width = int(image['height']), int(image['width'])

        ann = self.refer.refToAnn.get(ref['ref_id'], None)
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
                    # list of polygon lists
                    seg_iter = seg
            else:
                return None

            for seg_item in seg_iter:
                decoded = self._decode_mask(seg_item, height, width)
                decoded = self._align_mask_to_image_size(decoded, height, width)
                binary_mask = np.clip(binary_mask + decoded, 0, 1)

            if binary_mask.sum() == 0:
                return None
            return binary_mask.astype(np.uint8)
        except Exception:
            return None

    def _resolve_image_path(self, file_name: str) -> str:
        primary = os.path.join(self.refer.IMAGE_DIR, file_name)
        if os.path.exists(primary):
            return primary

        # RIS-LAD on some setups stores images in images/ris_lad/*.jpg without JPEGImages.
        if 'JPEGImages' in self.refer.IMAGE_DIR:
            fallback_dir = self.refer.IMAGE_DIR.replace('/JPEGImages', '')
            fallback = os.path.join(fallback_dir, file_name)
            if os.path.exists(fallback):
                return fallback

        # Keep original path for error visibility if no candidate exists.
        return primary

    def _format_prompt(self, text: str) -> str:
        if self.with_thinking:
            question = f'Please segment {text} in this image.'
            think_prompt = (
                'You should first think about the reasoning process in the mind and then provides '
                'the user with the answer. Please respond with segmentation mask in both the thinking '
                'process and the answer.'
            )
            template_prompt = (
                'The reasoning process and answer are enclosed within <think> </think> and '
                '<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> '
                '<answer> answers here </answer>.'
            )
            return f'<image>\n{question}\n\n{think_prompt}\n\n{template_prompt}'

        return f'<image>\n Please segment {text} in this image.'

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        return {
            'index': idx,
            'ref_id': sample['ref_id'],
            'image_id': sample['image_id'],
            'image_path': sample['image_path'],
            'raw_text': sample['raw_text'],
            'text': sample['text'],
            'image': image,
            'gt_mask': sample['gt_mask'],
        }


def _to_hw_mask(mask):
    mask = np.asarray(mask)
    if mask.ndim == 3:
        if mask.shape[0] == 1:
            mask = mask[0]
        elif mask.shape[-1] == 1:
            mask = mask[..., 0]
        else:
            mask = mask[0]
    return (mask > 0).astype(np.uint8)


def _select_prediction_mask(pred_text: str, pred_masks):
    if pred_masks is None or len(pred_masks) == 0:
        return None

    cleaned_pred_text = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()
    _, answer_seg_idx = find_seg_indices(cleaned_pred_text)

    if len(answer_seg_idx) == 0:
        all_seg_count = len(re.findall(r'\[SEG\]', cleaned_pred_text))
        if all_seg_count > 0:
            answer_seg_idx = list(range(min(all_seg_count, len(pred_masks))))

    if len(answer_seg_idx) > 0:
        selected = []
        for idx in answer_seg_idx:
            if idx < len(pred_masks):
                selected.append(_to_hw_mask(pred_masks[idx]))
        if len(selected) > 0:
            out = selected[0].copy()
            for m in selected[1:]:
                out = np.logical_or(out, m)
            return out.astype(np.uint8)

    selected = [_to_hw_mask(m) for m in pred_masks]
    out = selected[0].copy()
    for m in selected[1:]:
        out = np.logical_or(out, m)
    return out.astype(np.uint8)


def _safe_iou(pred_mask, gt_mask):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    inter = int((pred & gt).sum())
    union = int((pred | gt).sum())
    iou = float(inter) / float(union) if union > 0 else 0.0
    return iou, inter, union


def _compute_metrics(valid: List[Dict], thresholds: List[float]) -> Dict:
    ious = np.array([x['iou'] for x in valid], dtype=np.float64) if len(valid) > 0 else np.array([], dtype=np.float64)
    total_inter = int(np.sum([x['inter'] for x in valid])) if len(valid) > 0 else 0
    total_union = int(np.sum([x['union'] for x in valid])) if len(valid) > 0 else 0

    m_iou_ratio = float(np.mean(ious)) if len(ious) > 0 else 0.0
    o_iou_ratio = float(total_inter) / float(total_union) if total_union > 0 else 0.0

    pr_ratio = {}
    pr_percent = {}
    for t in thresholds:
        key = f'Pr@{t:.1f}'
        val = float(np.mean(ious >= t)) if len(ious) > 0 else 0.0
        pr_ratio[key] = val
        pr_percent[key] = round(100.0 * val, 2)

    metrics = {
        'num_samples': len(valid),
        'thresholds': thresholds,
        'ratio': {
            'mIoU': m_iou_ratio,
            'oIoU': o_iou_ratio,
            **pr_ratio,
        },
        'percent': {
            'mIoU': round(100.0 * m_iou_ratio, 2),
            'oIoU': round(100.0 * o_iou_ratio, 2),
            **pr_percent,
        },
        'total_inter': total_inter,
        'total_union': total_union,
    }
    return metrics


def _resolve_device(device: str) -> str:
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def _resolve_eval_targets(dataset_arg: str, split_arg: str):
    datasets = list(DATASET_MAP.keys()) if dataset_arg == 'all' else [dataset_arg]
    splits = ['val', 'test'] if split_arg == 'all' else [split_arg]
    return datasets, splits


def parse_args():
    parser = argparse.ArgumentParser(description='Sa2VA ROS-RefSeg evaluator with Pr@k, mIoU, oIoU')
    parser.add_argument('model_path', nargs='?', default=None, help='HF model path for Sa2VA')
    parser.add_argument('--eval_model_path', type=str, default=None, help='Alias for model_path')
    parser.add_argument('--dataset', choices=['RRSISD', 'ris_lad', 'all'], default='all')
    parser.add_argument('--split', choices=['val', 'test', 'all'], default='all')
    parser.add_argument('--data_root', type=str, default='./data/ROS-Sa2VA')
    parser.add_argument('--output_root', type=str, default='./work_dirs/ros_refseg_eval')
    parser.add_argument('--run_name', type=str, default='')
    parser.add_argument('--use_think', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_step', type=int, default=0)
    parser.add_argument('--save_pred_mask', action='store_true')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')

    args = parser.parse_args()

    if args.model_path is None and args.eval_model_path is None:
        raise ValueError('Please provide model path using positional model_path or --eval_model_path.')
    if args.model_path is None:
        args.model_path = args.eval_model_path

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def _run_single_eval(
    model,
    tokenizer,
    processor,
    dataset_name: str,
    split: str,
    args,
    run_dir: str,
    rank: int,
    world_size: int,
):
    dataset = ROSRefSegEvalDataset(
        data_root=args.data_root,
        option=dataset_name,
        split=split,
        with_thinking=args.use_think,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )

    pred_mask_dir = os.path.join(run_dir, f'pred_masks_{dataset_name}_{split}')
    if args.save_pred_mask:
        os.makedirs(pred_mask_dir, exist_ok=True)

    results = []
    max_step = args.max_step if args.max_step > 0 else None

    for idx, data_batch in enumerate(tqdm.tqdm(dataloader, total=len(dataset), desc=f'{dataset_name}-{split}')):
        if max_step is not None and idx >= max_step:
            break

        model_inputs = {
            'image': data_batch['image'],
            'text': data_batch['text'],
        }

        try:
            pred = model.predict_forward(**model_inputs, tokenizer=tokenizer, processor=processor)
        except TypeError as e:
            if 'unexpected keyword argument' in str(e) and 'processor' in str(e):
                pred = model.predict_forward(**model_inputs, tokenizer=tokenizer)
            else:
                raise

        pred_text = pred.get('prediction', '')
        pred_masks = pred.get('prediction_masks', [])

        selected_mask = _select_prediction_mask(pred_text, pred_masks)
        gt_mask = _to_hw_mask(data_batch['gt_mask'])

        if selected_mask is None:
            selected_mask = np.zeros_like(gt_mask, dtype=np.uint8)

        iou, inter, union = _safe_iou(selected_mask, gt_mask)

        pred_mask_path = None
        if args.save_pred_mask:
            pred_mask_path = os.path.join(pred_mask_dir, f'{rank}_{idx}_{data_batch["ref_id"]}.png')
            Image.fromarray((selected_mask * 255).astype(np.uint8)).save(pred_mask_path)

        results.append(
            {
                'index': int(data_batch['index']),
                'ref_id': int(data_batch['ref_id']),
                'image_id': int(data_batch['image_id']),
                'image_path': data_batch['image_path'],
                'text': data_batch['raw_text'],
                'prediction': pred_text,
                'iou': float(iou),
                'inter': int(inter),
                'union': int(union),
                'pred_area': int((selected_mask > 0).sum()),
                'gt_area': int((gt_mask > 0).sum()),
                'pred_mask_path': pred_mask_path,
                'dataset': dataset_name,
                'split': split,
            }
        )

    tmpdir = os.path.join(
        run_dir,
        f'dist_tmp_{dataset_name}_{split}_{os.path.basename(args.model_path).replace("/", "_")}',
    )
    gathered = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)

    if get_rank() != 0:
        return None

    if max_step is not None:
        gathered = gathered[:max_step]

    valid = [x for x in gathered if x is not None]
    metrics = _compute_metrics(valid, PR_THRESHOLDS)

    metrics.update(
        {
            'dataset': dataset_name,
            'split': split,
            'model_path': args.model_path,
            'data_root': args.data_root,
            'use_think': args.use_think,
            'format_note': 'percent metrics are xx.xx values (0-100 scale)',
        }
    )

    metrics_path = os.path.join(run_dir, f'metrics_{dataset_name}_{split}.json')
    details_path = os.path.join(run_dir, f'details_{dataset_name}_{split}.jsonl')

    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    with open(details_path, 'w', encoding='utf-8') as f:
        for item in valid:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(json.dumps(metrics['percent'], ensure_ascii=False, indent=2))
    print(f'Saved: {metrics_path}')
    print(f'Saved: {details_path}')

    return {
        'dataset': dataset_name,
        'split': split,
        'metrics_path': metrics_path,
        'details_path': details_path,
        'num_samples': metrics['num_samples'],
        'percent': metrics['percent'],
    }


def main():
    args = parse_args()

    datasets, splits = _resolve_eval_targets(args.dataset, args.split)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_tag = os.path.basename(args.model_path.rstrip('/'))

    run_name = args.run_name or f'sa2va_ros_refseg_{model_tag}_{timestamp}'
    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = _resolve_device(args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA but no GPU is available. Use --device cpu.')

    if args.launcher != 'none':
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        if device == 'cuda':
            torch.cuda.set_device(rank)
    else:
        rank, world_size = 0, 1

    if rank == 0:
        config_path = os.path.join(run_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        print(f'Run directory: {run_dir}')
        if device == 'cpu':
            print('Warning: CPU mode enabled. Sa2VA inference can be very slow.')

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=(device == 'cuda'),
        trust_remote_code=True,
    ).eval()
    if device == 'cuda':
        model = model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True) if 'qwen' in args.model_path.lower() else None

    summary = []
    for dataset_name in datasets:
        for split in splits:
            out = _run_single_eval(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                dataset_name=dataset_name,
                split=split,
                args=args,
                run_dir=run_dir,
                rank=rank,
                world_size=world_size,
            )
            if rank == 0 and out is not None:
                summary.append(out)

    if rank == 0:
        summary_path = os.path.join(run_dir, 'summary_all.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f'Saved: {summary_path}')


if __name__ == '__main__':
    main()
