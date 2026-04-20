# Research Plan: Are Induction Heads a Large Source of Leakage?

## Motivation & Novelty Assessment

### Why This Research Matters
Induction heads are one of the clearest mechanistic candidates for in-context copying, so if they also drive unwanted copying when the task requires randomness or non-reuse, they become a concrete target for controlling leakage-like behavior. This matters for privacy, robustness, and deployment settings where models should avoid echoing salient prompt content unless it is explicitly relevant.

### Gap in Existing Work
The local literature review shows strong evidence that induction heads support in-context learning and may contribute to repetition curse, but no local paper directly tests whether these heads increase copying pressure on prompts where copying is undesired. Existing leakage and memorization work also suggests the phenomenon may be distributed, so a direct causal test is still missing.

### Our Novel Contribution
We will run a head-level causal study connecting induction-like circuits to two behaviors in the same real model: intended copying on a controlled copy task and unintended copying on prompts that explicitly ask for a random answer. The key novelty is measuring whether ablating empirically identified induction heads reduces repeated-context bias and increases output entropy relative to matched random-head controls.

### Experiment Justification
- Experiment 1: Identify candidate induction heads in `gpt2-small` using a prefix-matching attention score on repeated-token synthetic sequences. This is needed to avoid relying only on literature claims about head identity.
- Experiment 2: Measure copy-task accuracy with no intervention, induction-head ablation, and random-head ablation. This validates that the identified heads really mediate copying in our setup.
- Experiment 3: Measure repeated-context bias and entropy on prompts that ask for a random answer from a constrained candidate set. This directly tests the user hypothesis about leakage into non-copying tasks.
- Experiment 4: Run sensitivity checks across several random seeds and prompt templates. This is needed to separate a robust circuit-level effect from prompt-specific artifacts.

## Research Question
Do induction heads in a real transformer materially increase unintended copying pressure on prompts that should instead yield diffuse or random outputs?

## Background and Motivation
Prior work in the local review establishes that induction heads are a major mechanism for in-context pattern matching, and newer work links them to repetition pathologies. At the same time, work on diffuse distributions shows that language models often struggle to output high-entropy answers when the task requires randomness. The missing link is a direct causal test: whether induction-style circuits measurably push probability mass toward prompt-repeated items even when the instruction says not to copy.

## Hypothesis Decomposition
- H1: A small subset of heads in `gpt2-small` will score highly on induction-style prefix matching.
- H2: Ablating those heads will reduce performance on a controlled copy task more than ablating the same number of random heads.
- H3: On random-choice prompts with repeated distractors, the full model will assign excess probability to the repeated distractor relative to a uniform target.
- H4: Induction-head ablation will reduce that excess repeated-token probability and increase normalized entropy more than random-head ablation.
- Alternative explanation A: The effect is a generic consequence of reducing model capacity, not induction-specific circuitry.
- Alternative explanation B: The effect is mostly lexical priors from the output layer rather than in-context copying.
- Alternative explanation C: The effect is prompt-template specific and not robust across random seeds or wording.

## Proposed Methodology

### Approach
Use TransformerLens on `gpt2-small` to identify heads that preferentially attend from the second occurrence of a repeated token to its earlier occurrence under repeated-sequence prompts. Then compare three conditions across tasks: full model, top induction-head ablation, and random-head ablation. Evaluate both intended copying and unintended copying pressure on random-answer prompts using exact-answer metrics and logit-distribution metrics.

### Experimental Steps
1. Load `gpt2-small` in TransformerLens with fixed seeds and CUDA.
   Rationale: provides direct access to attention patterns and head-level interventions.
2. Construct repeated-token synthetic prompts over a small candidate vocabulary and score each head by how strongly it attends to previous matching tokens.
   Rationale: identifies candidate induction heads empirically in this run.
3. Select the top-k induction heads and matched random-head sets.
   Rationale: enables a causal comparison against a non-specific ablation control.
4. Evaluate next-token accuracy on held-out repeated-sequence copy prompts.
   Rationale: confirms the identified heads matter for copying behavior.
5. Build random-answer prompts with repeated distractors from a constrained option set.
   Rationale: isolates whether prompt repetition biases the model when the instruction demands randomness.
6. For each prompt, record the repeated distractor probability, target-set entropy, top-1 output, and repeated-token selection rate under each intervention.
   Rationale: captures both distributional leakage pressure and realized copying behavior.
7. Repeat the random-head control and prompt generation across multiple seeds.
   Rationale: checks robustness and supports statistical testing.

### Baselines
- Full model with no ablation.
- Random-head ablation control with the same number of heads as the induction set.
- Uniform target distribution over candidate answers for the random-answer task.
- Simple lexical baseline: prompts without repeated distractor context, to measure background preference for each option.

### Evaluation Metrics
- Induction score per head: mean attention paid to prior matching token position.
- Copy-task accuracy: fraction of prompts where the correct repeated token is the top predicted next token.
- Repeated distractor probability: softmax mass on the repeated context item at the answer position.
- Repeated-token selection rate: fraction of prompts where the repeated distractor is top-1.
- Normalized entropy: entropy over the constrained candidate set divided by `log(n_candidates)`.
- KL divergence to uniform over the candidate set.
- Effect size: Cohen's d for key pairwise comparisons.

### Statistical Analysis Plan
- Primary paired comparison: induction-head ablation vs full model on repeated distractor probability and normalized entropy.
- Secondary paired comparison: induction-head ablation vs random-head ablation on the same metrics.
- Use paired t-tests when normality of paired differences is acceptable by Shapiro-Wilk; otherwise use Wilcoxon signed-rank.
- Report 95% confidence intervals by bootstrap, uncorrected and Benjamini-Hochberg corrected p-values for the main metric family.
- Significance threshold: `alpha = 0.05`.

## Expected Outcomes
Results support the hypothesis if: (1) identified induction heads clearly matter on the copy task, and (2) ablating them reduces repeated-distractor probability while increasing entropy on random-answer prompts beyond the random-head control. Results weaken the hypothesis if random-head and induction-head ablation behave similarly or if copy-task effects do not transfer to the random-answer setting.

## Timeline and Milestones
1. Planning and environment verification: 20 minutes.
2. Dependency installation and model-loading smoke tests: 20 minutes.
3. Implementation of scoring, ablation, and evaluation pipeline: 60 minutes.
4. Experiment runs and figure generation: 60 minutes.
5. Statistical analysis, documentation, and validation: 40 minutes.

## Potential Challenges
- `gpt2-small` may show weaker instruction-following on random-answer prompts than instruction-tuned models.
  Mitigation: constrain candidate sets and evaluate logits directly rather than relying only on sampled generations.
- Head identification may be noisy.
  Mitigation: use aggregate induction scores over many prompts and compare against known literature patterns.
- Head ablation may degrade performance globally.
  Mitigation: use random-head ablation and no-repeat-context controls to isolate induction-specific effects.
- Tokenization may split some candidate strings.
  Mitigation: use single-token candidates only.

## Success Criteria
- The pipeline runs end-to-end reproducibly in the local `.venv`.
- At least one head set shows a clear induction score separation from the background.
- Induction-head ablation meaningfully reduces copy-task accuracy versus full model.
- On random-answer prompts, induction-head ablation lowers repeated-token bias and raises normalized entropy relative to the full model and preferably relative to random-head controls.
- `REPORT.md` documents actual results, figures, statistical tests, limitations, and reproduction commands.
