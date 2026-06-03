# internvl2.5-1b 30% budgets
# ablation: CAS lambda: fixed 3
from mmengine.hooks import (
    CheckpointHook,
    DistSamplerSeedHook,
    IterTimerHook,
    LoggerHook,
    ParamSchedulerHook,
)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from peft import LoraConfig

from projects.sa2va.models.compression_method import (
    TransformerScorer,
    ScheduledWeightHook,
)
from projects.sa2va.models import (
    Sa2VAModelDev,
    SAM2TrainRunner,
    DirectResize,
    InternVLMLLM_Train_Dev,
)
from projects.sa2va.datasets import (
    sa2va_collect_fn,
    Sa2VA01RefSeg,
    LLaVADataset,
    Sa2VA03RefVOS,
    Sa2VA04VideoQA,
    Sa2VA05GCGDataset,
    Sa2VA06VPDataset,
)

from projects.sa2va.datasets.data_utils import ConcatDatasetSa2VA

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
# hf download OpenGVLab/InternVL2_5-1B --local-dir ./pretrained/InternVL2_5-1B
# pretrained_pth：微调用，需要下载hf格式，然后用自带tools/convert_to_pth.py转换得到
path = "./pretrained/sa2va-base/InternVL2_5-1B"
pretrained_pth = "./pretrained/sa2va-models/Sa2VA-1B.pth"

# Data
template = "qwen_chat"
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

# Scheduler & Optimizer
batch_size = 2  # per_device
accumulative_counts = 4  # on 16 gpus
dataloader_num_workers = 8
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 2000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

special_tokens = ["[SEG]", "<p>", "</p>", "<vp>", "</vp>"]

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side="right",
)

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)
#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=Sa2VAModelDev,
    training_bs=batch_size,
    special_tokens=special_tokens,
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVLMLLM_Train_Dev,
        use_flash_attn=True,
        model_path=path,
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            modules_to_save=["embed_tokens", "lm_head"],
        ),
        # 896维，中间已经映射到了llm的维度
        # 其他参数默认
        importance_scorer=dict(
            type=TransformerScorer,
            in_features=896,
        ),
        # 预算，只拿30%的最重要特征
        budgets=0.3,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss, use_sigmoid=True, reduction="mean", loss_weight=2.0
    ),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction="mean",
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5,
    ),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = "./data/baseline/"
VIDEO_DATA_ROOT = DATA_ROOT + "video_datas/"

# this is for datasets with masks
sa2va_default_dataset_configs = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    prompt_template=prompt_template,
    max_length=max_length,
)

# this is for datasets without masks
sa2va_qa_default_dataset_configs = dict(
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    prompt_template=prompt_template,
    max_length=max_length,
)

######################### ImageRefSeg ##################################
RES_ROOT = DATA_ROOT + 'ref_seg/'
sa2va_data_01_refseg_configs = [
    dict(
        type=Sa2VA01RefSeg,
        name='RefCOCO',
        data_root=RES_ROOT + 'refcoco',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(unc).p',
        num_classes_per_sample=5,
        repeats=5,
        **sa2va_default_dataset_configs,
    ),
    dict(
        type=Sa2VA01RefSeg,
        name='RefCOCO+',
        data_root=RES_ROOT + 'refcoco+',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(unc).p',
        num_classes_per_sample=5,
        repeats=5,
        **sa2va_default_dataset_configs,
    ),
    dict(
        type=Sa2VA01RefSeg,
        name='RefCOCOg',
        data_root=RES_ROOT + 'refcocog',
        data_prefix=dict(img_path='coco2014/train2014/'),
        ann_file='instances.json',
        split_file='refs(umd).p',
        num_classes_per_sample=5,
        repeats=4,
        **sa2va_default_dataset_configs,
    ),
]


######################### VideoRefSeg ##################################
sa2va_data_03_refvos_configs = [
    dict(
        type=Sa2VA03RefVOS,
        name='ReVOS',
        image_folder=VIDEO_DATA_ROOT + 'revos/',
        expression_file=VIDEO_DATA_ROOT + 'revos/' + 'meta_expressions_train_.json',
        mask_file=VIDEO_DATA_ROOT + 'revos/' + 'mask_dict.json',
        repeats=10,
        dataset_type='default',
        **sa2va_default_dataset_configs
    ),
    dict(
        type=Sa2VA03RefVOS,
        name='MeVIS',
        image_folder=VIDEO_DATA_ROOT + 'mevis/train/JPEGImages',
        expression_file=VIDEO_DATA_ROOT + 'mevis/train/meta_expressions.json',
        mask_file=VIDEO_DATA_ROOT + 'mevis/train/mask_dict.json',
        repeats=4,
        dataset_type='default',
        **sa2va_default_dataset_configs
    ),
    dict(
        type=Sa2VA03RefVOS,
        name='RefYTVOS',
        image_folder=VIDEO_DATA_ROOT + 'rvos/train/JPEGImages/',
        expression_file=VIDEO_DATA_ROOT + 'rvos/meta_expressions/train/meta_expressions.json',
        mask_file=VIDEO_DATA_ROOT + 'rvos/mask_dict.pkl',
        repeats=4,
        dataset_type='refytvos',
        **sa2va_default_dataset_configs
    ),
    dict(
        type=Sa2VA03RefVOS,
        name='Ref-SAV',
        image_folder=VIDEO_DATA_ROOT + 'sam_v_full/',
        expression_file=VIDEO_DATA_ROOT + 'Ref-SAV.json',
        repeats=4,
        dataset_type='refsav',
        **sa2va_default_dataset_configs
    )
]

train_dataset = dict(
    type=ConcatDatasetSa2VA,
    datasets=[
        *sa2va_data_01_refseg_configs,
        *sa2va_data_03_refvos_configs,
    ],
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=sa2va_collect_fn),
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype="bfloat16",
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True,
    ),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True,
    ),
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=ScheduledWeightHook, log_interval=10, fixed=3.0),
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit,
    ),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)

# set visualizer
visualizer = None

# set log level
log_level = "INFO"

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
