# Sa2VA Dev HF Conversion + DAVIS17 Evaluation Commands

This document records the exact commands to reproduce:
1. Convert trained PTH to HF format.
2. Run DAVIS17 inference with 4 GPUs.
3. Compute final J / F / J&F metrics.

## 0) Preconditions

- Workspace root: /hy-tmp/Sa2VA
- Python env: /hy-tmp/Sa2VA/.venv
- Trained checkpoint:
  - work_dirs/sa2va_in25_1b_dev_debug/iter_436.pth
- Training config:
  - projects/sa2va/configs/sa2va_in25_1b_dev_debug.py
- DAVIS17 data root (already prepared):
  - /hy-tmp/Sa2VA/data/debug/video_datas/davis17

## 1) Convert PTH -> HF (Dev)

Run in project root:

```bash
source /hy-tmp/Sa2VA/.venv/bin/activate
PYTHONPATH=/hy-tmp/Sa2VA \
python tools/convert_to_hf_dev.py \
  projects/sa2va/configs/sa2va_in25_1b_dev_debug.py \
  work_dirs/sa2va_in25_1b_dev_debug/iter_436.pth \
  --save-path work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev
```

Expected key logs:
- Mapped state_dict keys: ...
- All keys matched successfully!
- Save the hf model into work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev

HF output folder:
- work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev

## 2) Optional Smoke Test (Small DAVIS subset)

Use this to quickly confirm eval path before full run:

```bash
source /hy-tmp/Sa2VA/.venv/bin/activate
PYTHONPATH=/hy-tmp/Sa2VA \
python projects/sa2va/evaluation/sa2va_eval_ref_vos.py \
  work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev \
  --dataset DAVIS \
  --data_root /hy-tmp/Sa2VA/data/debug \
  --work_dir work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis_smoke \
  --num_workers 0 \
  --max_samples 3
```

Smoke output JSON:
- work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis_smoke/DAVIS/results.json

## 3) Full DAVIS17 Inference with 4 GPUs

```bash
source /hy-tmp/Sa2VA/.venv/bin/activate
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash projects/sa2va/evaluation/dist_test.sh \
  projects/sa2va/evaluation/sa2va_eval_ref_vos.py \
  work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev \
  4 \
  --dataset DAVIS \
  --data_root /hy-tmp/Sa2VA/data/debug \
  --work_dir work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis \
  --num_workers 4
```

Full inference output JSON:
- work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis/DAVIS/results.json

## 4) Compute DAVIS17 J / F / J&F

```bash
source /hy-tmp/Sa2VA/.venv/bin/activate
PYTHONPATH=/hy-tmp/Sa2VA \
python tools/eval/eval_davis.py \
  work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis/DAVIS/results.json \
  --mevis_exp_path /hy-tmp/Sa2VA/data/debug/video_datas/davis17/meta_expressions/valid/meta_expressions.json \
  --mevis_mask_path /hy-tmp/Sa2VA/data/debug/video_datas/davis17/valid/mask_dict.pkl \
  --save_name davis17_val.json
```

Metric output file:
- work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis/DAVIS/davis17_val.json

## 5) Check Final Metrics

```bash
cat /hy-tmp/Sa2VA/work_dirs/sa2va_in25_1b_dev_debug/iter_436_hf_dev_eval_davis/DAVIS/davis17_val.json
```

Example output from current run:

```json
{
  "J": 64.04,
  "F": 72.83,
  "J&F": 68.43
}
```

## 6) Notes

- Keep PYTHONPATH=/hy-tmp/Sa2VA for conversion and metric scripts.
- DAVIS metric script is CPU multiprocessing based; inference acceleration mainly comes from multi-GPU Step 3.
- The HF eval script now supports local fallback loading for this converted local model path.
