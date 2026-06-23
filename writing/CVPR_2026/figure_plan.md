# Figure Plan

## 1. Motivation Figure
- Goal: show why unified dense grounding needs token selection.
- Message: long visual streams are expensive; redundant tokens can be pruned.
- Place: end of intro opening paragraph.
- Needed: one schematic comparing baseline token load vs selected tokens.

## 2. Framework Figure
- Goal: show Sa2VA-Select pipeline.
- Message: fused image/video + VP stream, LIS, DiffTopK, hard Top-K, [SEG] to SAM2.
- Place: intro main figure.
- Needed: your revised framework artwork.

## 3. Method Mechanism Figure
- Goal: explain how the selector works step by step.
- Message: training path vs inference path, scoring, soft mask, hard mask, CAS.
- Place: method section.
- Needed: a detailed diagram, possibly with two sub-panels.

## 4. Video Qualitative Figure A
- Goal: show the `distractor-dominant confusion` failure mode.
- Message: baseline is pulled toward visually dominant but irrelevant content; ours stays closer to the referred target.
- Place: experiments qualitative subsection.
- Recommended cases:
  - `revos_case_pack/MOSE__train__c28d81a9/exp_56`
  - `revos_case_pack/LV-VIS__train__02698/exp_14`
- Layout suggestion: 2 cases, each with 3 frames and 4 rows (`Frame / Baseline / Ours / GT`).
- Selected frames:
  - `exp_56`: `00008`, `00014`, `00021`
  - `exp_14`: `00005`, `00010`, `00018`
- Caption draft:
  - `Figure A. Qualitative examples of distractor-dominant confusion on ReVOS. In the striped-tail cat case and the left-side dog case, the Sa2VA-1B baseline is easily pulled toward larger or more visually salient regions, even when they are not the referred target. Sa2VA-Select stays closer to the ground truth by suppressing distractor-heavy visual evidence and preserving cues that are more relevant to fine-grained identity and relational grounding.`

## 5. Video Qualitative Figure B
- Goal: show the `early mis-tracking / occlusion-induced drift` failure mode.
- Message: once baseline locks onto the wrong target or drifts under occlusion, later tracking remains unstable; ours is more temporally stable.
- Place: experiments qualitative subsection.
- Recommended cases:
  - `mevis_case_pack/6eac00b5f389/exp_9`
  - `mevis_case_pack/f610df51c78a/exp_12`
  - optional alternative/additional case: `revos_case_pack/OVIS__train__5ba84e3f/exp_15`
- Layout suggestion: 2 cases in the main paper; move the optional third case to appendix if space is tight.
- Selected frames:
  - `exp_9`: `00010`, `00018`, `00031`
  - `exp_12`: `00019`, `00035`, `00057`
- Caption draft:
  - `Figure B. Qualitative examples of early mis-tracking and occlusion-induced drift on MeViS. When the target is initially weakly visible, changes pose, or becomes partially occluded, the Sa2VA-1B baseline tends to lock onto an incorrect object and remain unstable in later frames. Sa2VA-Select maintains more consistent target evidence over time, producing masks that better follow the intended object throughout the sequence.`

## 6. Token Retention Figure
- Goal: show what is kept and what is discarded.
- Message: selector keeps semantically useful regions.
- Place: experiments or analysis subsection.
- Needed: heatmap / retained-token visualization / discarded-token overlay.

## 7. Trade-off Curve Figure
- Goal: show performance vs retention / efficiency vs retention.
- Message: moderate retention is the best operating region.
- Place: experiments main analysis.
- Needed: line plot or combined curve plot from current tables.

## 8. Ablation Figure/Table
- Goal: show LIS / DiffTopK / CAS contributions.
- Message: each component matters.
- Place: experiments ablation subsection.
- Status: main ablation table is now available from the 30% budget experiments.
- Current data trend: fixed-weight variants underperform; annealed CAS recovers performance; `0.1~3` currently gives the best overall Avg and ReVOS.
- Recommended visual companion: a compact line/bar plot over CAS schedule strength (`no CAS`, `fixed 3`, `0.1~1`, `0.1~2`, `0.1~3`) to emphasize that annealing, not merely stronger regularization, drives the gain.

## 9. Table 2 Update
- Status: main `MeViS` column has now been restored in Table 2 using the latest user-provided results.
- Core video column: `Ref-DAVIS17` remains a key video segmentation metric in the summary table.
- Follow-up: if a cleaner standard-split note is needed later, keep the main-table wording and efficiency-table wording explicitly separated.
- Note: MMMU, AI2D, MMStar, SQAtest are locally re-evaluated Sa2VA-1B baselines.
