from typing import Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor


from projects.sa2va.models.sa2va import Sa2VAModel
from projects.sa2va.models.compression_method import TransformerScorer


class Sa2VAModelDev(Sa2VAModel):
    def __init__(
        self,
        mllm,
        tokenizer,
        grounding_encoder,
        loss_mask=None,
        loss_dice=None,
        torch_dtype=torch.bfloat16,
        pretrained_pth=None,
        frozen_sam2_decoder=True,
        special_tokens=None,
        loss_sample_points=False,
        num_points=12544,
        template=None,
        # for arch selection
        arch_type: Literal["intern_vl", "qwen", "llava"] = "intern_vl",
        # ext
        # preprocessor=None,
        # bs
        training_bs: int = 0,
    ):
        # 基线初始化
        super().__init__(
            mllm,
            tokenizer,
            grounding_encoder,
            loss_mask,
            loss_dice,
            torch_dtype,
            pretrained_pth,
            frozen_sam2_decoder,
            special_tokens,
            loss_sample_points,
            num_points,
            template,
            arch_type,
            training_bs,
        )

    pass
