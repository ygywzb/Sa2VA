# ROS-RefSeg Evaluation Toolkit (Sa2VA)

This toolkit is outside ReSaP and focuses on two separated parts:

1. Evaluation and metric computation
2. Post-evaluation visualization from saved artifacts

## Directory

- `eval_sa2va_ros_refseg.py`: model inference + metric computation + artifact saving
- `compute_metrics_from_details.py`: recompute metrics from `details_*.jsonl` only (no model/GPU)
- `visualize_cases.py`: qualitative case visualization (high/low/mid/mixed)
- `plot_metrics.py`: IoU histogram, Pr@k curve, cumulative mIoU curve
- `check_mask_consistency.py`: compare GT mask parsing consistency (training-style vs legacy REFER.getMask)

## Metrics covered

For each dataset/split, outputs include:

- `Pr@0.5`, `Pr@0.6`, `Pr@0.7`, `Pr@0.8`, `Pr@0.9`
- `mIoU`
- `oIoU`

Each metric is saved in two forms:

- `ratio`: 0~1
- `percent`: `xx.xx` (0~100 scale), matching paper-style table display

## Run evaluation

```bash
source /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/.venv/bin/activate
cd /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA:$PYTHONPATH
export CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}

python projects/sa2va/evaluation/ros_refseg/eval_sa2va_ros_refseg.py \
  /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/pretrained/sa2va-models/Sa2VA-1B \
  --dataset all \
  --split all \
  --data_root /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/data/ROS-Sa2VA \
  --output_root /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/work_dirs/ros_refseg_eval \
  --run_name sa2va_1b_rosrefseg \
  --save_pred_mask
```

### Notes for current no-GPU situation

You can still run metric recomputation and visualization from existing details:

```bash
python projects/sa2va/evaluation/ros_refseg/compute_metrics_from_details.py \
  --details_path work_dirs/eval_sa2va1b_200/details_RRSISD_test.jsonl
```

## Output artifact design

A run directory contains:

- `config.json`
- `summary_all.json`
- `metrics_{dataset}_{split}.json`
- `details_{dataset}_{split}.jsonl`
- `pred_masks_{dataset}_{split}/` (if enabled)

This keeps all per-sample data for downstream visualizations (top IoU, low IoU, mid IoU, keyword filtering, etc.).

## Visualizations

### 1) Cases (high/low/mid/mixed)

```bash
python projects/sa2va/evaluation/ros_refseg/visualize_cases.py \
  --run_dir work_dirs/ros_refseg_eval/sa2va_1b_rosrefseg \
  --dataset RRSISD \
  --split test \
  --mode high \
  --num_cases 6
```

### 2) Metric plots

```bash
python projects/sa2va/evaluation/ros_refseg/plot_metrics.py \
  --run_dir work_dirs/ros_refseg_eval/sa2va_1b_rosrefseg \
  --dataset RRSISD \
  --split test
```

## GT Mask Consistency Check

This script verifies whether the evaluator's robust mask parsing stays consistent
with legacy REFER parsing on sampled refs.

```bash
python projects/sa2va/evaluation/ros_refseg/check_mask_consistency.py \
  --dataset RRSISD \
  --split test \
  --data_root /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/data/ROS-Sa2VA \
  --sample_size 500 \
  --output_root /home/lzjd/DaiMa/Sa2VA/ygy/Sa2VA/work_dirs/ros_refseg_eval
```

## Suggested paper-table workflow

1. Run `eval_sa2va_ros_refseg.py` for `RRSISD/ris_lad` and `val/test`.
2. Read each `metrics_*.json` and extract `percent` values into your table.
3. Use `visualize_cases.py` and `plot_metrics.py` for qualitative and statistical analysis.
