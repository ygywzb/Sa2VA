# 针对Sa2VA-1B模型的demo
# BaseMLLM: InternVL2.5-1B, LLM: Qwen2.5-0.5B-Instruct
import argparse
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoConfig

import torch

import cv2

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def get_rank_and_world_size():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


def split_model(model_path):
    import math

    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    print(f"Model {model_path} has {num_layers} layers.")

    # Since the first GPU will be used for ViT, treat it as 0.5 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    print(f"num_layers_per_gpu: {num_layers_per_gpu}")

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = (
                rank + world_size * i
            )
            layer_cnt += 1

    device_map["vision_model"] = rank
    device_map["mlp1"] = rank
    device_map["language_model.model.tok_embeddings"] = rank
    device_map["language_model.model.embed_tokens"] = rank
    device_map["language_model.output"] = rank
    device_map["language_model.model.norm"] = rank
    device_map["language_model.lm_head"] = rank
    device_map[f"language_model.model.layers.{num_layers - 1}"] = rank
    device_map["grounding_encoder"] = rank
    device_map["text_hidden_fcs"] = rank

    return device_map


def parse_args():
    parser = argparse.ArgumentParser(description="Video Reasoning Segmentation")
    parser.add_argument("--image_folder", default=None, help="Path to image file")
    parser.add_argument("--model_path", default="ByteDance/Sa2VA-8B")
    parser.add_argument("--work-dir", default=None, help="The dir to save results.")
    parser.add_argument(
        "--text", type=str, default="<image>Please describe the video content."
    )
    parser.add_argument(
        "--single_image", default=None, help="Path to single image file."
    )
    parser.add_argument("--select", type=int, default=-1)
    parser.add_argument(
        "--use_flash_attn",
        type=bool,
        default=True,
        help="Whether to use flash attention.",
    )
    args = parser.parse_args()
    return args


def visualize(pred_mask, image_path, work_dir):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors="g", alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(work_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)


if __name__ == "__main__":
    print(torch.cuda.is_available())

    cfg = parse_args()
    model_path = cfg.model_path

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        use_flash_attn=cfg.use_flash_attn,
        dtype="auto",
        trust_remote_code=True,
    )

    model = model.cuda()
    print(model.device)

    # For distributed inference, uncomment the following lines to get device_map
    # device_map=split_model(model_path)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_path,
    #     use_flash_attn=cfg.use_flash_attn,
    #     torch_dtype=torch.bfloat16,
    #     device_map=device_map,
    #     trust_remote_code=True
    # )

    assert (
        "qwen" not in model_path.lower()
    ), "This demo is only for Sa2VA-1B model with InternVL2.5-1B base model."

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    image_files = []
    image_paths = []
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    if cfg.single_image is not None:
        filename = os.path.basename(cfg.single_image)
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_files.append(filename)
            image_paths.append(cfg.single_image)
        else:
            raise ValueError("The specified single_image is not a valid image file.")
    else:
        assert (
            cfg.image_folder is not None
        ), "Please specify image_folder when single_image is not given."
        for filename in sorted(list(os.listdir(cfg.image_folder))):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(filename)
                image_paths.append(os.path.join(cfg.image_folder, filename))

    vid_frames = []
    # 读图片
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        vid_frames.append(img)

    # 指定图片文件路径
    if cfg.single_image is not None:
        cfg.select = 1

    if cfg.select > 0:
        img_frame = vid_frames[cfg.select - 1]

        print(f"Selected frame {cfg.select}")
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            image=img_frame,
            text=cfg.text,
            tokenizer=tokenizer,
        )  # type: ignore
    else:
        print(f"The input is:\n{cfg.text}")
        result = model.predict_forward(
            video=vid_frames,
            text=cfg.text,
            tokenizer=tokenizer,
        )  # type: ignore

    prediction = result["prediction"]
    print(f"The output is:\n{prediction}")

    if "[SEG]" in prediction and Visualizer is not None:
        _seg_idx = 0
        pred_masks = result["prediction_masks"][_seg_idx]
        for frame_idx in range(len(vid_frames)):
            # -------------------ADD-------------------
            # 单帧模式，循环只会运行一次
            if cfg.select > 0 and frame_idx != 0:
                break
            # -------------------ADD-------------------
            pred_mask = pred_masks[frame_idx]
            # -------------------ADD-------------------
            # 单帧模式，只处理指定帧
            if cfg.select > 0:
                frame_idx = cfg.select - 1
            # -------------------ADD-------------------

            if cfg.work_dir:
                os.makedirs(cfg.work_dir, exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], cfg.work_dir)
            else:
                os.makedirs("./temp_visualize_results", exist_ok=True)
                visualize(pred_mask, image_paths[frame_idx], "./temp_visualize_results")
    else:
        pass
