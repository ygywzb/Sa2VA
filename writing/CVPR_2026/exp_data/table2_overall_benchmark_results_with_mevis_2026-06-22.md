# Table 2 Overall Benchmark Results with MeViS

Source: user-provided experiment screenshot on 2026-06-22.
Purpose: canonical local record for the updated main summary table in `writing/CVPR_2026/sec/4_experiments.tex`.

## Notes
- This version restores the main `MeViS` column in Table 2.
- `Ref-DAVIS17` remains a core video segmentation metric in the summary table.
- `AI2D`, `MMStar`, `MMMU`, and `SQA^test` are still locally re-evaluated Sa2VA-1B baselines, but they do not appear in this summary table.

## Data

| Method | RefCOCO | RefCOCO+ | RefCOCOg | MeViS | Ref-YTVOS | Ref-DAVIS17 | ReVOS | MME | MMBench | SEED-Bench | Video-MME | Avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Sa2VA-1B (100%) | 77.4 | 69.9 | 72.3 | 41.7 | 65.3 | 72.3 | 47.6 | 1381/405 | 68.3 | 64.8 | 39.9 | 100% |
| Sa2VA-Select-1B (40%) | 77.5 | 69.1 | 74.8 | 44.9 | 66.2 | 71.0 | 51.2 | 1403/384 | 66.1 | 69.3 | 40.4 | 102% |
| Sa2VA-Select-1B (30%) | 77.4 | 69.6 | 74.6 | 43.5 | 67.0 | 70.6 | 51.3 | 1388/399 | 66.8 | 68.0 | 41.4 | 102% |
| Sa2VA-Select-1B (20%) | 75.7 | 66.1 | 72.3 | 44.0 | 66.5 | 69.3 | 50.2 | 1357/365 | 64.1 | 66.1 | 38.9 | 99.2% |
| Sa2VA-Select-1B (10%) | 76.1 | 65.8 | 72.7 | 42.5 | 65.2 | 68.0 | 48.1 | 1345/333 | 63.0 | 63.7 | 36.3 | 96.9% |
