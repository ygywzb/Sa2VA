# Ablation at 30% Budget

Source: user-provided experiment screenshot on 2026-06-22.
Purpose: canonical local record for the ablation subsection/table added to `writing/CVPR_2026/sec/4_experiments.tex`.

## Notes
- All ablation runs use the same visual token budget / retention setting: `30%`.
- `baseline` refers to `Sa2VA-1B (100%)` without token selection.
- `ours` corresponds to the current full `Sa2VA-Select-1B` setting with `LIS + DiffTopK + CAS` and annealed `\lambda_t = 0.1 \sim 3`.

## Data

| Config | LIS + DiffTopK | CAS | lambda_t | RefCOCO | RefCOCO+ | RefCOCOg | DAVIS | ReVOS | Avg |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | - | - | - | 77.4 | 69.9 | 72.3 | 72.3 | 47.6 | 100% |
| 1 | sqrt | - | 1 (fixed) | 75.2 | 64.7 | 71.6 | 70.5 | 48.6 | 97.7% |
| 2 | sqrt | sqrt | 3 (fixed) | 73.7 | 62.5 | 68.7 | 68.8 | 48.4 | 95.3% |
| 3 | sqrt | sqrt | 0.1~1 | 77.4 | 68.7 | 74.7 | 69.2 | 50.1 | 100.5% |
| 4 | sqrt | sqrt | 0.1~2 | 76.1 | 66.4 | 72.1 | 70.6 | 49.4 | 98.9% |
| ours | sqrt | sqrt | 0.1~3 | 77.4 | 69.6 | 74.6 | 70.6 | 51.3 | 101.6% |
