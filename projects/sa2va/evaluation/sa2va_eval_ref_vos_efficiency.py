import argparse
import os
import types
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="RefVOS efficiency-only evaluation")
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
        local_modeling_file = Path(model_path) / "modeling_sa2va_dev_chat.py"
        if not local_modeling_file.exists():
            raise err

        from projects.sa2va.hf.models.modeling_sa2va_dev_chat import Sa2VADevChatModel

        model = Sa2VADevChatModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval().cuda()
        return model


def apply_efficiency_monkey_patch(model):
    from projects.sa2va.hf.models.modeling_sa2va_dev_chat_efficiency import (
        Sa2VADevChatEfficiencyModel,
    )

    if hasattr(model, "_original_generate"):
        return model

    model._original_generate = model.generate
    model.generate = types.MethodType(Sa2VADevChatEfficiencyModel.generate, model)
    return model


def summarize_metrics(all_metrics):
    valid_metrics = [m for m in all_metrics if isinstance(m, dict)]

    prefill = [
        float(m["generation_prefill_time_ms"])
        for m in valid_metrics
        if m.get("generation_prefill_time_ms") is not None
    ]
    latency = [
        float(m["generation_latency_time_ms"])
        for m in valid_metrics
        if m.get("generation_latency_time_ms") is not None
    ]
    memory = [
        int(m["after_generation_memory"])
        for m in valid_metrics
        if m.get("after_generation_memory") is not None
    ]
    visual_num = [
        int(m["input_visual_token_number"])
        for m in valid_metrics
        if m.get("input_visual_token_number") is not None
    ]

    summary = {
        "samples": len(valid_metrics),
        "avg_prefill_ms": (sum(prefill) / len(prefill)) if prefill else None,
        "avg_latency_ms": (sum(latency) / len(latency)) if latency else None,
        "avg_max_memory_gb": ((sum(memory) / len(memory)) / (1024**3)) if memory else None,
        "avg_visual_token_num": (sum(visual_num) / len(visual_num)) if visual_num else None,
    }
    return summary


def print_summary(summary):
    print("===== Efficiency Summary =====")
    print(f"Samples: {summary['samples']}")

    if summary["avg_max_memory_gb"] is not None:
        print(f"Average max memory: {summary['avg_max_memory_gb']} GB")
    else:
        print("Average max memory: N/A")

    if summary["avg_prefill_ms"] is not None:
        print(f"Average prefill time: {summary['avg_prefill_ms']} mSces")
    else:
        print("Average prefill time: N/A")

    if summary["avg_latency_ms"] is not None:
        print(f"Average latency: {summary['avg_latency_ms']} mSces")
    else:
        print("Average latency: N/A")

    if summary["avg_visual_token_num"] is not None:
        print(f"Average visual token num: {summary['avg_visual_token_num']}")
    else:
        print("Average visual token num: N/A")


if __name__ == "__main__":
    args = parse_args()

    if "EVAL_TIME" not in os.environ:
        os.environ["EVAL_TIME"] = "True"

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
    model = apply_efficiency_monkey_patch(model)

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

    local_metrics = []
    for idx, item in enumerate(tqdm.tqdm(dataloader, disable=(rank != 0))):
        if args.max_samples > 0 and idx >= args.max_samples:
            break

        with torch.no_grad():
            model.predict_forward(
                video=item["images"],
                text=item["text_prompt"],
                tokenizer=tokenizer,
                processor=processor,
            )

        local_metrics.append(getattr(model, "_last_efficiency_metrics", None))

    expected_total = len(dataset) if args.max_samples < 0 else min(args.max_samples, len(dataset))
    all_metrics = collect_results_cpu(local_metrics, expected_total)

    if get_rank() == 0:
        summary = summarize_metrics(all_metrics)
        print_summary(summary)

    if rank == 0:
        print("Done")
