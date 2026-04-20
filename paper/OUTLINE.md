## Title
- Induction Heads Modulate Repeated-Token Bias but Are Not a Singular Source of Leakage in GPT-2 Small

## Abstract
- Problem: test whether induction-like heads cause leakage-like copying on prompts that ask for random choice.
- Gap: prior work connects induction heads to in-context learning and repetition, but not directly to random-choice leakage.
- Method: identify top induction-like heads in GPT-2 small, ablate them with TransformerLens, compare against random-head controls on copy, random-choice, and WikiText utility tasks.
- Main findings: L5H5/L5H1 reduce repeated-token bias by 0.0285 but also reduce entropy by 0.0417 and raise WikiText perplexity from 89.9 to 169.7; sensitivity runs flip the sign when L6H9 is included.
- Significance: induction-like heads matter, but the effect is heterogeneous and not a clean leakage-specific target.

## Introduction
- Hook: copying circuits are attractive mitigation targets only if they causally explain unwanted copying.
- Importance: privacy, robustness, and sampling control all depend on separating useful in-context reuse from undesirable prompt-conditioned repetition.
- Gap: no direct causal test of induction heads on prompts that should yield diffuse/random outputs.
- Approach: head detection plus ablation on copy and random-choice tasks in one real model; refer to overview figure.
- Quantitative preview: top heads L5H5/L5H1; bias delta drops from 0.0496 to 0.0211, entropy drops from 0.860 to 0.819, perplexity rises from 89.9 to 169.7; sensitivity sign flip across head subsets.
- Contributions:
  - We identify induction-like heads in GPT-2 small with empirical scoring.
  - We test the same intervention on intended and unintended copying tasks.
  - We show the strongest ablation result trades lower bias for lower entropy and worse language modeling.
  - We show sensitivity runs reverse the bias effect, arguing against a monolithic leakage circuit story.

## Related Work
- Theme 1: induction heads and in-context learning (Olsson et al.; Crosbie and Shutova; Ren et al.).
- Theme 2: induction heads and repetition pathologies (Wang et al.).
- Theme 3: memorization and extraction are distributed phenomena (Lee et al.; Yu et al.; Ippolito et al.; Huang et al.; Ishihara).
- Theme 4: diffuse/random output failures (Zhang et al.).
- Positioning: unlike prior work, combine head-level intervention with random-choice leakage metric and utility check in same study.

## Methodology
- Formalize repeated-token bias delta and normalized entropy metrics.
- Model/setup: GPT-2 small, TransformerLens, CPU due to CUDA mismatch.
- Head detection prompts over six color tokens; select top-k induction-like heads.
- Three conditions: full, induction ablation, random-head ablation.
- Tasks: copy task, random-choice leakage task, WikiText utility.
- Statistical analysis: Wilcoxon and bootstrap CI due to non-normal paired differences.

## Results
- Figure with head scores.
- Main table: copy accuracy, bias delta, entropy, perplexity across conditions.
- Statistical table for final run comparisons.
- Sensitivity table for head subsets.
- Interpretation:
  - top detected heads align with canonical mid-layer induction heads.
  - L5H5/L5H1 lower bias but also lower entropy and hurt utility.
  - Sensitivity runs with L6H9 reverse the bias effect.

## Discussion
- Main answer: induction-like heads matter but are heterogeneous, not singular leakage source.
- Explain surprising copy-task increase in final run and instability across subsets.
- Discuss entanglement between reducing repeated-token bias and global distribution sharpening / utility loss.
- Limitations: one model, synthetic task, CPU-constrained sample sizes, no extraction attack.
- Broader implications: mechanistic interventions need behavior-specific and utility-aware evaluation.

## Conclusion
- Summarize contributions and core takeaway.
- Emphasize sensitivity sign flip as strongest empirical result.
- Future work: instruction-tuned models, soft descaling, extraction benchmarks.

## Tables/Figures Plan
- Figure 1: head score plot (`figures/head_scores.png`)
- Figure 2: copy accuracy plot (`figures/copy_accuracy.png`)
- Figure 3: random bias and entropy plot (`figures/random_bias_entropy.png`)
- Figure 4: WikiText perplexity plot (`figures/wikitext_perplexity.png`)
- Table 1: final main metrics
- Table 2: paired statistical tests
- Table 3: sensitivity runs
