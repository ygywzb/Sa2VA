import argparse
import gc
import inspect
import os
import types
from pathlib import Path

import numpy as np
import torch
import torch.distributed
import torch.utils.data
import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from projects.sa2va.evaluation.dataset import RefVOSDataset
from projects.sa2va.evaluation.utils import (
    _init_dist_pytorch,
    _init_dist_slurm,
    collect_results_cpu,
    get_dist_info,
    get_rank,
)


DATASETS_INFO = {
    "DAVIS": {
        "data_root": "data/video_datas/davis17/",
        "image_folder": "data/video_datas/davis17/valid/JPEGImages/",
        "expression_file": "data/video_datas/davis17/meta_expressions/valid/meta_expressions.json",
        "mask_file": "data/video_datas/davis17/valid/mask_dict.pkl",
    },
    "MEVIS": {
        "data_root": "data/video_datas/mevis/valid/",
        "image_folder": "data/video_datas/mevis/valid/JPEGImages",
        "expression_file": "data/video_datas/mevis/valid/meta_expressions.json",
        "mask_file": None,
    },
    "MEVIS_U": {
        "data_root": "data/video_datas/mevis/valid_u/",
        "image_folder": "data/video_datas/mevis/valid_u/JPEGImages",
        "expression_file": "data/video_datas/mevis/valid_u/meta_expressions.json",
        "mask_file": "data/video_datas/mevis/valid_u/mask_dict.json",
    },
    "MEVIS_T": {
        "data_root": "data/video_datas/mevis/test/",
        "image_folder": "data/video_datas/mevis/test/JPEGImages",
        "expression_file": "data/video_datas/mevis/test/meta_expressions_release.json",
        "mask_file": None,
    },
    "REFYTVOS": {
        "data_root": "data/video_datas/rvos/",
        "image_folder": "data/video_datas/rvos/valid/JPEGImages/",
        "expression_file": "data/video_datas/rvos/meta_expressions/valid/meta_expressions.json",
        "mask_file": None,
    },
    "REVOS": {
        "data_root": "data/video_datas/revos/",
        "image_folder": "data/video_datas/revos/",
        "expression_file": "data/video_datas/revos/meta_expressions_valid_.json",
        "mask_file": None,
    },
    "REF_SAV": {
        "data_root": "data/ref_sav_eval/",
        "image_folder": "data/ref_sav_eval/videos",
        "expression_file": "data/ref_sav_eval/meta_expressions_valid.json",
        "mask_file": "data/ref_sav_eval/mask_dict.json",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="RefVOS visualize-only evaluation (dev)")
    parser.add_argument("model_path", help="hf model path.")
    parser.add_argument(
        "--dataset",
        choices=DATASETS_INFO.keys(),
        default="MEVIS_U",
        help="Specify a dataset",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--data_root", default="./data", help="Root directory for all datasets.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_dir", default=None, help="Optional directory to save importance maps.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Only evaluate the first N samples for smoke validation; -1 means full dataset.",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def load_model_with_fallback(model_path):
    try:
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval().cuda()
        return model
    except FileNotFoundError as err:
        model_dir = Path(model_path)
        dev_modeling = model_dir / "modeling_sa2va_dev_chat.py"
        base_modeling = model_dir / "modeling_sa2va_chat.py"

        if dev_modeling.exists():
            try:
                from projects.sa2va.hf.models.modeling_sa2va_dev_chat import Sa2VADevChatModel
            except ModuleNotFoundError:
                repo_root = Path(__file__).resolve().parents[3]
                if str(repo_root) not in os.sys.path:
                    os.sys.path.append(str(repo_root))
                from projects.sa2va.hf.models.modeling_sa2va_dev_chat import Sa2VADevChatModel

            model = Sa2VADevChatModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval().cuda()
            return model

        if base_modeling.exists():
            try:
                from projects.sa2va.hf.models.modeling_sa2va_chat import Sa2VAChatModel
            except ModuleNotFoundError:
                repo_root = Path(__file__).resolve().parents[3]
                if str(repo_root) not in os.sys.path:
                    os.sys.path.append(str(repo_root))
                from projects.sa2va.hf.models.modeling_sa2va_chat import Sa2VAChatModel

            model = Sa2VAChatModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).eval().cuda()
            return model

        raise err


def apply_visualize_monkey_patch(model):
    from projects.sa2va.hf.models.modeling_sa2va_dev_chat_visualize import (
        Sa2VADevChatVisualizeModel,
    )

    if hasattr(model, "_original_predict_forward"):
        return model

    model._original_predict_forward = model.predict_forward
    model._original_generate = model.generate

    model.predict_forward = types.MethodType(
        Sa2VADevChatVisualizeModel.predict_forward, model
    )
    model.generate = types.MethodType(Sa2VADevChatVisualizeModel.generate, model)
    model._store_importance_grid = types.MethodType(
        Sa2VADevChatVisualizeModel._store_importance_grid, model
    )
    model._build_visual_token_mappings = types.MethodType(
        Sa2VADevChatVisualizeModel._build_visual_token_mappings, model
    )
    model._build_importance_maps = types.MethodType(
        Sa2VADevChatVisualizeModel._build_importance_maps, model
    )

    return model


def build_predict_forward_kwargs(model, item, tokenizer, processor):
    sig = inspect.signature(model.predict_forward)
    params = sig.parameters

    kwargs = {
        "video": item["images"],
        "text": item["text_prompt"],
    }

    if "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    if "processor" in params and processor is not None:
        kwargs["processor"] = processor

    return kwargs


if __name__ == "__main__":
    args = parse_args()

    for _, info in DATASETS_INFO.items():
        for path_key, path_val in info.items():
            if path_val is not None and (
                "folder" in path_key or "file" in path_key or "root" in path_key
            ):
                info[path_key] = os.path.join(
                    args.data_root, os.path.relpath(path_val, "./data")
                )

    if args.launcher == "none":
        rank = 0
        world_size = 1
    elif args.launcher == "pytorch":
        import datetime

        _init_dist_pytorch("nccl", timeout=datetime.timedelta(minutes=30))
        rank, world_size = get_dist_info()
    elif args.launcher == "slurm":
        _init_dist_slurm("nccl")
        rank, world_size = get_dist_info()
    else:
        raise ValueError(f"Unsupported launcher: {args.launcher}")

    model = load_model_with_fallback(args.model_path)
    model = apply_visualize_monkey_patch(model)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if "qwen" in args.model_path.lower():
        processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    else:
        processor = None

    dataset_info = DATASETS_INFO[args.dataset]
    dataset = RefVOSDataset(
        image_folder=dataset_info["image_folder"],
        expression_file=dataset_info["expression_file"],
        mask_file=dataset_info["mask_file"],
    )

    sampler = torch.utils.data.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda x: x[0],
    )

    local_results = []
    for idx, item in enumerate(tqdm.tqdm(dataloader, disable=(rank != 0))):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        with torch.no_grad():
            predict_kwargs = build_predict_forward_kwargs(
                model=model,
                item=item,
                tokenizer=tokenizer,
                processor=processor,
            )
            output = model.predict_forward(**predict_kwargs)

        if args.save_dir and get_rank() == 0 and output is not None:
            os.makedirs(args.save_dir, exist_ok=True)
            if "importance_maps" in output:
                maps = output["importance_maps"]
                maps_array = np.stack(maps, axis=0) if isinstance(maps, list) else np.asarray(maps)
                payload = {"importance_maps": maps_array}
                if "importance_tile_map" in output:
                    payload["importance_tile_map"] = output["importance_tile_map"]
                if "importance_thumbnail_map" in output:
                    payload["importance_thumbnail_map"] = output["importance_thumbnail_map"]
                save_path = os.path.join(args.save_dir, f"importance_{idx:06d}.npz")
                np.savez_compressed(save_path, **payload)

        if output is not None:
            output.pop("importance_maps", None)
            output.pop("importance_tile_map", None)
            output.pop("importance_thumbnail_map", None)
        local_results.append(output)
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    expected_total = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    all_results = collect_results_cpu(local_results, expected_total)

    if get_rank() == 0:
        print(f"Done, collected {len(all_results)} results")

    if rank == 0:
        print("Done")
