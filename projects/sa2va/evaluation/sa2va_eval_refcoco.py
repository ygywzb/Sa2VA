import argparse
import copy
import json
import os
import re
from datetime import datetime
from pathlib import Path
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np

from transformers import AutoModel, AutoTokenizer, AutoProcessor

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from dataset import RESDataset

from projects.sa2va.models.utils import find_seg_indices

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--vis',
        action='store_true',
        help='whether to visualize the results')
    parser.add_argument(
        '--with-thinking',
        action='store_true',
        help='whether to enable the reasoning process')
    
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--deepspeed', type=str, default=None) # dummy
    parser.add_argument('--data_root', default='/mnt/bn/zilongdata-us/xiangtai/Sa2VA/data', help='Root directory for all datasets.')
    parser.add_argument(
        '--metric_output_dir',
        type=str,
        default=None,
        help='Directory to save final evaluation metric as a JSON file. If not set, metric will not be written to disk.')
    parser.add_argument(
        '--metric_output_name',
        type=str,
        default=None,
        help='Optional output filename for metric JSON. If not set, an auto-generated name is used.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
    'grefcoco': {'splitBy': "unc", 'dataset_name': 'grefcoco'},
}

IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def extract_answer_seg_indices(text: str):
    """Return SEG indices for answer span; fallback to all [SEG] when no answer tags exist."""
    all_seg_indices = [m.start() for m in re.finditer(r'\[SEG\]', text)]
    if len(all_seg_indices) == 0:
        return []

    _, answer_seg_idx = find_seg_indices(text)
    if len(answer_seg_idx) > 0:
        return answer_seg_idx

    # Some checkpoints answer directly without <answer> tags.
    return list(range(len(all_seg_indices)))


def save_metric_to_file(metric: dict, args):
    """Save final metric dictionary to a JSON file when output directory is provided."""
    if args.metric_output_dir is None:
        return None

    dataset_output_dir = os.path.join(args.metric_output_dir, args.dataset)
    os.makedirs(dataset_output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = args.metric_output_name or f'{args.dataset}_{args.split}_metric.json'
    output_path = os.path.join(dataset_output_dir, filename)

    payload = {
        'model_path': args.model_path,
        'dataset': args.dataset,
        'split': args.split,
        'with_thinking': args.with_thinking,
        'metric': metric,
        'timestamp': timestamp,
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path




def main():
    args = parse_args()

    image_folder = os.path.join(args.data_root, 'glamm_data/images/coco2014/train2014/')
    data_path = os.path.join(args.data_root, 'ref_seg/')

    if args.launcher != 'none':
        import datetime
        _init_dist_pytorch('nccl', timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    try:
        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
    except FileNotFoundError as err:
        model_path = Path(args.model_path)
        dev_modeling = model_path / 'modeling_sa2va_dev_chat.py'
        base_modeling = model_path / 'modeling_sa2va_chat.py'
        if dev_modeling.exists():
            from projects.sa2va.hf.models.modeling_sa2va_dev_chat import Sa2VADevChatModel
            model = Sa2VADevChatModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval().cuda()
        elif base_modeling.exists():
            from projects.sa2va.hf.models.modeling_sa2va_chat import Sa2VAChatModel
            model = Sa2VAChatModel.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval().cuda()
        else:
            raise err

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    if 'qwen' in args.model_path.lower():
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = None
    #model.preparing_for_generation(tokenizer, max_new_tokens=512)

    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset = RESDataset(
        image_folder=image_folder,
        dataset_name=dataset_info['dataset_name'],
        data_path=data_path,
        split=args.split,
        reasoning=args.with_thinking,
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False,
        drop_last=False
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
        collate_fn=lambda x:x[0],
    )

    results = []
    cnt = 0
    model_name = args.model_path.strip('/').split('/')[-1]
    for data_batch in tqdm.tqdm(dataloader):
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        texts = data_batch['text']
        #print("texts:", texts)
        img_metas = {'img_id': data_batch['img_id'], 
                     'image_path': data_batch['image_path']}
        del data_batch['img_id'], data_batch['gt_masks'], data_batch['image_path'], data_batch['text']
        pred_masks = []
        pred_texts = []
        pred_n_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred = model.predict_forward(**_data_batch, tokenizer=tokenizer, processor=processor)
            pred_mask = pred['prediction_masks']
            pred_text = pred['prediction']
            print(f"Text: {text}")
            print(f"Predicted Text: {pred_text}")
            
            print(f"Number of pred masks: {len(pred_mask)}")

            cleaned_pred_text = pred_text.replace('<|im_end|>', '').replace('<|end|>', '').strip()
            answer_seg_idx = extract_answer_seg_indices(cleaned_pred_text)
            if len(answer_seg_idx) == 0:
                answer_seg_idx = extract_answer_seg_indices(pred_text)
            print(f"Answer seg indices with fallback parser: {answer_seg_idx}")
            
            pred_texts.append(pred_text)
            
            # whether the prediction is empty
            if len(pred_mask) == 0:
                #print('No mask predicted')
                pred_masks.append(None)
                pred_n_masks.append(None)
                continue
            else:
                # List (1, h, w) -> (n, h, w)
                pred_n_masks.append(np.concatenate(pred_mask, axis=0))

                if len(answer_seg_idx) > 0:
                    print(f"Found {len(answer_seg_idx)} [SEG] tokens in answer, using corresponding masks")

                    selected_masks = [pred_mask[idx] for idx in answer_seg_idx if idx < len(pred_mask)]
                    if len(selected_masks) == 0 and len(pred_mask) > 0:
                        # Keep one prediction mask when SEG appears but index alignment fails.
                        selected_masks = [pred_mask[0]]
                    
                    if len(selected_masks) > 0:
                        final_mask = selected_masks[0]
                        for mask in selected_masks[1:]:
                            final_mask = final_mask | mask
                        _ret_mask = mask_to_rle(final_mask)
                        pred_masks.append(_ret_mask)
                    else:
                        print("No valid masks for answer [SEG] tokens")
                        pred_masks.append(None)
                else:
                    print("No [SEG] token found in answer")
                    pred_masks.append(None)
                    continue

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)
    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        metric = dataset.evaluate(results)
        print(metric)
        metric_file = save_metric_to_file(metric, args)
        if metric_file is not None:
            print(f"Metric saved to: {metric_file}")

if __name__ == '__main__':
    main()
