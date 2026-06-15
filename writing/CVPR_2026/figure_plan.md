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
- Goal: compare baseline / ours / GT on video segmentation dataset 1.
- Message: ours stays close to GT or fixes baseline mistakes.
- Place: experiments qualitative subsection.
- Needed: 3-5 representative cases with same frames and masks.

## 5. Video Qualitative Figure B
- Goal: same as above on video segmentation dataset 2.
- Message: generalization across datasets.
- Place: experiments qualitative subsection.
- Needed: another set of representative cases.

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
- Needed: ablation numbers or a small table.

## 9. Table 2 Update
- Current rule: temporarily remove main MeViS column.
- Core video column: Ref-DAVIS17.
- Need: later add MeViS back once the standard split is run.
- Note: MMMU, AI2D, MMStar, SQAtest are locally re-evaluated Sa2VA-1B baselines.
