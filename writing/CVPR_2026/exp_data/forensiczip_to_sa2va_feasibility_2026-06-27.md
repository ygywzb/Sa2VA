ForensicZip -> Sa2VA baseline feasibility note
Date: 2026-06-27

Source files inspected:
- writing/CVPR_2026/paper_pdf/Lai et al. - 2026 - ForensicZip More Tokens are Better but Not Necessary in Forensic Vision-Language Models.pdf
- writing/CVPR_2026/exp_data/forensiczip_extract_2026-06-27.txt
- writing/CVPR_2026/sec/3_method.tex

Bottom-line judgment
- ForensicZip is not a clean direct replacement for the current Sa2VA-Select route.
- It is only partially compatible with Sa2VA, and mainly as an inference-time, training-free video-token pruning heuristic.
- It is not suitable as a strong main-method direction for unified dense grounded understanding unless you are willing to substantially weaken the claim and accept likely risk on image tasks, visual-prompt tokens, and segmentation fidelity.

Why it is attractive
- It is explicitly training-free and uses physical Top-K pruning.
- The paper states that it "seamlessly integrates into existing forensic MLLMs via physical pruning" and keeps only a compressed token sequence before the language model.
- For video, it scores tokens using inter-frame transport novelty plus a high-frequency prior, then keeps top-K patch tokens while always keeping the global token.

Why it does not cleanly fit Sa2VA
1. Task mismatch:
- ForensicZip is built for multimedia forensics, not dense grounding or referring segmentation.
- Its main assumption is that useful evidence lives in semantically weak but artifact-heavy regions.
- Sa2VA needs spatially precise object evidence, relational cues, and prompt-conditioned evidence for [SEG] generation.

2. Token-type mismatch:
- Sa2VA consumes a unified visual stream that includes both ordinary image/video tokens and VP-conditioned tokens.
- ForensicZip assumes a standard per-frame patch-token stream plus one global token.
- The method does not define how VP-conditioned tokens should be scored, protected, or jointly budgeted.

3. Video-specific scoring core:
- The strongest part of ForensicZip is the Birth-Death OT score over adjacent frames.
- That mechanism naturally applies to video patch tokens, but not to image-only inputs.
- The paper provides a single-image fallback based on distance to the frame mean embedding, but that fallback is weakly motivated for dense grounding and does not use referring text or prompt information.

4. Frequency prior may hurt segmentation:
- ForensicZip boosts tokens with strong Laplacian/high-frequency response.
- In referring segmentation, many targets or useful context regions are not high-frequency.
- The heuristic may over-select edges, textured distractors, or noisy backgrounds while under-selecting smooth but semantically crucial target regions.

5. No text/prompt conditioning:
- The scoring is intentionally anti-semantic and not query-aware.
- Sa2VA often depends on expression grounding such as left/right, first appearing object, or interaction-specific targets.
- A text-agnostic scorer is risky because the kept tokens may be visually anomalous but irrelevant to the referring expression.

Recommended conclusion
- Do not position ForensicZip as a new main paper direction on top of Sa2VA unless you only want a very limited "training-free heuristic pruning baseline" story.
- If you still want to try it, the safest framing is:
  "Can a training-free video novelty heuristic compress Sa2VA visual tokens at inference time without retraining?"
- That is a much weaker and more defensible claim than replacing the current method with a new main contribution.

Most feasible integration plan
- Insert pruning at the same pre-LLM point as the current selector: after Sa2VA forms the unified visual tokens and before the MLLM.
- Apply ForensicZip-style scoring only to ordinary video/image tokens, not to VP-conditioned tokens.
- Always keep all VP-conditioned tokens.
- Always keep global/thumbnail anchor tokens.
- For video tokens:
  use inter-frame OT novelty on visual patch tokens only.
- For image tokens:
  either do not prune them at all, or use only a very conservative fallback heuristic.
- Perform hard Top-K pruning only at inference.

Minimal implementation sketch
1. Split unified tokens into:
- visual patch/video tokens
- protected tokens (VP tokens, special/global tokens, thumbnail anchors)
2. For video patch tokens:
- compute projected token features
- compute adjacent-frame OT cost with dummy node
- compute per-token birth score and transport cost
- optionally multiply by a mild high-frequency prior
- rank within each frame
3. Keep top-K patch tokens per frame
4. Concatenate retained patch tokens with all protected tokens
5. Feed the shortened sequence to the MLLM unchanged

What would make it publishably weak
- No retraining advantage is real, but the method would still be a task-mismatched heuristic.
- Without query-aware selection, gains on grounded segmentation are uncertain.
- The method has no principled story for VP tokens, which are central in Sa2VA.
- The single-image fallback is especially weak for RefCOCO-style tasks.

Practical recommendation
- If you need a low-cost emergency route, use ForensicZip only as an additional training-free baseline or ablation-style comparison, not as the new paper's core innovation.
- If you want a replacement main direction, it is better to continue searching for a training-free method that is:
  query-aware,
  compatible with image and video,
  and able to preserve prompt-conditioned tokens explicitly.
