# Codex Handoff for CVPR_2026 Paper

## Purpose
This file summarizes the current writing, figure, and table decisions for the `writing/CVPR_2026` paper draft so a new Codex session on another machine can continue the work without relying on chat history.

## Paper Positioning
- The paper should **not** be framed as inventing LIS / DiffTopK / CAS from scratch.
- The safe and intended story is:
  - VisionSelector shows learnable visual token selection for VQA-style MLLMs.
  - This paper studies whether the same idea transfers to **Sa2VA**, a **unified dense grounded image-video model** that jointly supports segmentation and conversation.
  - The key contribution is **setting extension + compatible integration + empirical validation**.
- The main difficulty to emphasize:
  - Sa2VA uses a **unified visual stream** after combining image/video tokens and VP-conditioned tokens.
  - Token reduction must preserve both **QA/conversation ability** and **segmentation-sensitive `[SEG]` generation**.

## Current Key Terms
- Method name: `Sa2VA-Select`
- Scorer name in the latest figure/text: `UVT-LIS`
  - Meaning: `Unified Visual Token-oriented Learnable Importance Scorer`
- Training-time selection: `soft selection`
- Inference-time selection: `hard selection`

## Files Already Revised
- `writing/CVPR_2026/main.tex`
- `writing/CVPR_2026/sec/0_abstract.tex`
- `writing/CVPR_2026/sec/1_intro.tex`
- `writing/CVPR_2026/sec/2_related.tex`
- `writing/CVPR_2026/sec/3_method.tex`
- `writing/CVPR_2026/sec/4_experiments.tex`
- `writing/CVPR_2026/sec/5_conclusion.tex`
- `writing/CVPR_2026/figure_plan.md`

## Current Figure Decisions

### Main Framework Figure
- The user already drew a revised overall framework figure.
- It should show:
  - Sa2VA first producing **unified visual tokens**
  - `UVT-LIS` scoring on unified visual tokens
  - training-time `soft selection`
  - inference-time `hard selection`
  - downstream `QA task` and `segmentation task`
  - total loss with constraint loss and downstream loss
- It should **not** over-explain DiffTopK internals; those belong in a separate mechanism figure.

### Detailed Mechanism Figure
- Should focus only on:
  - input unified visual tokens `V in R^{N x D}`
  - `UVT-LIS` scoring -> `s in R^N`
  - `DiffTopK` -> `M_soft`
  - `Top-k` -> `M_hard`
  - `L_constraint = BCE(M_soft, M_hard)`
  - `L_total = L_downstream + lambda_t * L_constraint`
  - small visual indication of curriculum annealing (`lambda_t`)
- Do not redraw the whole Sa2VA pipeline in this figure.

## Current Table / Metric Decisions

### Main Results Table (`tab:overall_table6_summary`)
- Temporarily **remove the main MeViS column** from the summary table.
- Reason:
  - the currently available user results are on `MeViS (val_u)`, not the intended standard MeViS split for the main table.
- `Ref-DAVIS17` should remain a **core video segmentation column** in the main summary table.
- `ReVOS` is also kept in the main table.
- When proper MeViS results are ready later, the MeViS column can be added back.

### Locally Re-evaluated Sa2VA-1B Baselines
The following Sa2VA-1B baseline numbers are **not directly from the original Sa2VA paper** and were run locally by the user:
- `AI2D`
- `MMStar`
- `MMMU`
- `SQA^test`

The experiments text should make that explicit.

## Current Data Notes

### Main Cross-Task Summary Data
User provided current summary-style values for:
- `Sa2VA-1B (100%)`
- `ours(40%)`
- `ours(30%)`
- `ours(20%)`
- `ours(10%)`

The manuscript currently uses only part of these in the main summary table.
Before final polishing, all cross-references to these numbers should be re-checked.

### Ref-SAV
- The draft currently keeps the Ref-SAV validation table and text.
- These values were treated as current valid numbers in the previous session.

### Efficiency
- `MeViS (val_u)` is still used in the **efficiency analysis** section.
- This is separate from the decision to temporarily remove the main MeViS column in the cross-task summary table.

## Current Writing Constraints
- Avoid overclaiming novelty.
- Use language like:
  - `compatible extension`
  - `learnable token selection in unified dense grounding`
  - `favorable accuracy-efficiency trade-off`
  - `preserves joint segmentation-and-conversation capability`
- Avoid language that implies LIS / DiffTopK / CAS are original inventions of this paper.

## Pending Work

### High Priority
- Continue refining the main framework figure.
- Create the detailed mechanism figure for `UVT-LIS + soft/hard selection + CAS`.
- Re-check all experiment tables for consistency with the user's latest numbers.
- Add qualitative visualization figures:
  - video segmentation comparison on dataset A
  - video segmentation comparison on dataset B
  - retained vs discarded token visualization
- Add trade-off curve / statistics figure(s).

### Medium Priority
- Add ablation subsection once the user confirms available ablation results.
- Revisit captions and linked prose after figures are finalized.

## Recommended Startup Prompt for a New Local Codex Session
Ask the new Codex session to do this first:

1. Read `writing/CVPR_2026/CODEX_HANDOFF.md`
2. Read:
   - `writing/CVPR_2026/sec/1_intro.tex`
   - `writing/CVPR_2026/sec/3_method.tex`
   - `writing/CVPR_2026/sec/4_experiments.tex`
   - `writing/CVPR_2026/figure_plan.md`
3. Continue the paper-writing and figure-guidance task from that state

## Suggested Human Workflow
- Sync this repository to the Windows machine.
- Open the same project in Codex Desktop on Windows.
- Start the new session with:
  - "Please first read `writing/CVPR_2026/CODEX_HANDOFF.md` and continue the CVPR_2026 paper writing/figure planning task from there."

