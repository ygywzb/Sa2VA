## train:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist.sh train projects/sa2va/configs/dev/sa2va_in25_1b_bgt30.py 4
```

## convert to hf
```bash
PYTHONPATH=./ python tools/convert_to_hf_dev.py projects/sa2va/configs/dev/sa2va_in25_1b_bgt30.py work_dirs/sa2va_in25_1b_bgt30/iter_xxxx.pth --save-path work_dirs/sa2va_in25_1b_bgt30/iter_xxxx_hf
```

## eval(refseg)
```bash
# --- Evaluation Commands ---
# eval_configs = {
#     "RefCOCO": {  => cIoU
#         "script": os.path.join(base_path, "sa2va_eval_refcoco.py"),
#         "datasets": ["refcoco", "refcoco_plus", "refcocog"],
#         "split": "test"
#     },
#     "GCG": {      => AP50
#         "script": os.path.join(base_path, "sa2va_eval_gcg.py"),
#         "split": "val",
#         "metrics_script": os.path.join(base_path, "metrics_gcg.py"),  => 只保留了AP50计算
#     },
#     "RefVOS": {   => J&F
#         "script": os.path.join(base_path, "sa2va_eval_ref_vos.py"),
#         "datasets": ["DAVIS", "MEVIS_U", "REF_SAV"], => Ref-YTVOS 和 ReVOS 无 mask，先不算
#     }
# }
# 
# 
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=./ python projects/sa2va/evaluation/run_all_evals.py work_dirs/sa2va_in25_1b_bgt30/iter_xxxx_hf --data_root ./data/baseline --gpus 4
```
RefCOCO 和 GCG 会直接输出结果到命令行，REF-VOS 输出results.json

---

<!-- 计算J&F:
```bash

``` -->
