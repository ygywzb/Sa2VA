# Trade-off Curve Data for Origin 2026

This file explains how to use `tradeoff_curve_origin_2026.tsv` for the planned trade-off / analysis figure in the CVPR 2026 Sa2VA-Select paper.

## Source tables

- Main performance table in manuscript:
  - `writing/CVPR_2026/sec/4_experiments.tex`
  - Table label: `tab:overall_table6_summary`
- Efficiency table in manuscript:
  - `writing/CVPR_2026/sec/4_experiments.tex`
  - Table label: `tab:efficiency_mevis`

## Column meanings

- `retention_ratio_pct`:
  - visual token retention ratio in percent
  - use this as the default X axis in Origin
- `retention_ratio_decimal`:
  - same ratio in decimal form
- `avg_relative_perf_pct`:
  - overall relative average from the main summary table
  - baseline is `100.0`
- `seg_perf_relative_pct`:
  - relative segmentation performance from the efficiency table (`Perf.` column)
  - baseline is `100.0`
- `ref_davis17_jf`, `revos_jf`, `mevis_main_jf`:
  - main task performance from the overall summary table
- `mevis_val_u_jf`:
  - MeViS `(val_u)` used in the efficiency table
- `max_gpu_gb`, `prefill_ms`, `e2e_ms`:
  - efficiency metrics measured on MeViS `(val_u)`

## Recommended main figure for Origin 2026

Use a 2-panel figure.

### Panel A: Performance vs. Retention Ratio

Recommended X axis:
- `retention_ratio_pct`

Recommended Y series:
- `avg_relative_perf_pct`
- `revos_jf`

Optional alternative:
- replace `revos_jf` with `ref_davis17_jf` if you want the figure to align more strongly with the current main-paper video metric emphasis

Reason:
- `avg_relative_perf_pct` shows the overall operating region
- `revos_jf` or `ref_davis17_jf` shows the dense grounding behavior directly

### Panel B: Efficiency vs. Retention Ratio

Recommended X axis:
- `retention_ratio_pct`

Recommended Y series:
- `max_gpu_gb`
- `prefill_ms`

Optional:
- add `e2e_ms` as a third curve only if the panel remains readable

Reason:
- `max_gpu_gb` and `prefill_ms` show the clearest monotonic efficiency gains
- `e2e_ms` is informative but less monotonic, so it can clutter the main panel

## Recommended plotting settings in Origin 2026

- Sort X axis in descending order:
  - `100, 40, 30, 20, 10`
- Use line + symbol plots
- Highlight `30%` and `40%` as the preferred operating region
- If using dual-axis plotting, keep:
  - left Y axis for performance
  - right Y axis for efficiency
- If dual-axis looks crowded, use two separate panels instead

## Suggested figure message

The intended takeaway is:

- moderate retention (`30%` to `40%`) preserves the best overall dense-grounding quality
- lower retention improves memory and prefill cost more aggressively
- the best deployment point depends on whether accuracy or latency is prioritized

## Suggested caption draft

`Performance-efficiency trade-offs across visual token retention ratios for Sa2VA-Select-1B. Moderate retention ratios (30% to 40%) preserve the strongest dense-grounding performance, while lower retention further reduces GPU memory and prefill cost. This result shows that Sa2VA-Select exposes a controllable operating region rather than a single fixed compression point.`
