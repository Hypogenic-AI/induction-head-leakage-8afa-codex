# Are Induction Heads a Large Source of Leakage?

## 1. Executive Summary
I tested whether empirically identified induction-like heads in `gpt2-small` push the model toward copying a repeated prompt item when the prompt instead asks for a diffuse or random answer. The short answer is: not in any simple, dominant way. The same head family that clearly scores as induction-like does not yield a stable “ablate heads, remove leakage” story.

The strongest final run found that ablating two high-scoring heads (`L5H5`, `L5H1`) significantly reduced repeated-token bias on random-choice prompts, but it also reduced output entropy and substantially damaged WikiText perplexity. Sensitivity runs using `L5H5` plus `L6H9` moved the effect in the opposite direction, increasing repeated-token bias. The practical implication is that induction heads are involved in these behaviors, but they do not look like a single large causal source of leakage in this model.

## 2. Research Question & Motivation
### Hypothesis
Induction heads may leak patterned prompt content even when the model should not be copying, and that mechanism might contribute to the well-known difficulty LMs have in producing random or diffuse outputs.

### Why This Matters
If a small, identifiable circuit is a major source of unwanted copying, it becomes a plausible target for mitigation in privacy, robustness, and sampling-control work. The local literature review showed strong evidence linking induction heads to in-context pattern matching and some evidence linking them to repetition curse, but no direct causal test of leakage on “should-be-random” prompts.

### Gap Filled Here
This project connected head-level interventions to two behaviors in the same real model:
1. Induction-style copying on controlled prompts.
2. Repeated-context bias on prompts asking for a random choice from a constrained candidate set.

## 3. Experimental Setup
### Model and Tools
- Model: `gpt2-small`
- Mechanistic toolkit: TransformerLens (editable install from `code/TransformerLens/`)
- Core libraries: `torch 2.11.0+cu130`, `transformers 5.5.4`, `datasets 4.8.4`, `scipy 1.17.1`, `pandas 3.0.2`, `matplotlib 3.10.8`, `seaborn 0.13.2`
- Hardware detected: 4x NVIDIA RTX A6000 by `nvidia-smi`
- Effective runtime device: CPU only
  PyTorch could not use CUDA because the installed driver/runtime pair was too old for the local torch build.

### Local Resources Used
- `datasets/synthetic_copy_task/` for validating the local resource and documenting an available copy-style benchmark.
- `datasets/wikitext_103_v1/` for a lightweight utility check via perplexity.
- `literature_review.md` and `resources.md` for experiment design and baseline selection.

### Prompt Families
#### Head detection prompts
Synthetic repeated-token sequences over six single-token color candidates:
- `red`, `blue`, `green`, `yellow`, `black`, `white`

These prompts were scored with TransformerLens `detect_head(..., "induction_head")` to identify high-scoring induction-like heads.

#### Copy task
Prompts of the form:
- first segment: `A B C D`
- second segment prefix: `A B C`
- target next token: `D`

This is a standard induction-style next-token continuation pattern.

#### Random-choice leakage task
Prompts asked the model to choose one item from a six-item candidate set “with no preference” or “at random”, with and without repeated distractor context such as:
- `Previous answers:red,red,red`

For each prompt I measured how much probability mass the model assigned to the repeated distractor relative to the neutral version of the same candidate list.

### Conditions
- Full model
- Induction-head ablation
- Random-head ablation controls

Head ablation was implemented by zeroing the selected heads’ `hook_z` activations in TransformerLens.

### Metrics
- Mean induction score per head
- Copy-task top-1 accuracy
- Repeated-token bias delta
  `P(repeated | biased prompt) - P(repeated | neutral prompt)`
- Normalized entropy over the six valid candidates
- KL divergence to uniform
- WikiText negative log-likelihood and perplexity

### Statistical Plan
For the final run I used paired tests on prompt-level metrics. Because the paired differences were non-normal in the final run, I reported Wilcoxon signed-rank tests, bootstrap confidence intervals, and Benjamini-Hochberg adjusted p-values.

## 4. Results
### Main Final Run
Configuration:
- `--head-prompts 48`
- `--copy-prompts 96`
- `--random-prompts 96`
- `--wikitext-samples 12`
- `--top-k-heads 2`

Top detected heads:
- `L5H5` with score `0.740`
- `L5H1` with score `0.548`

Key outputs are saved in:
- `results/summary.json`
- `results/copy_task_summary.csv`
- `results/random_bias_prompt_level.csv`
- `results/wikitext_utility.csv`
- `figures/head_scores.png`
- `figures/copy_accuracy.png`
- `figures/random_bias_entropy.png`
- `figures/wikitext_perplexity.png`

### Main Quantitative Results

| Metric | Full | Induction Ablation | Random Ablation Mean |
|---|---:|---:|---:|
| Copy accuracy | 0.729 | 0.823 | 0.723 |
| Repeated-token bias delta | 0.0496 | 0.0211 | 0.0466 |
| Normalized entropy | 0.860 | 0.819 | 0.847 |
| WikiText perplexity | 89.9 | 169.7 | 106.5 |

### Final-Run Statistical Tests

| Comparison | Mean diff | 95% bootstrap CI | Test | p | BH-adjusted p | Effect size |
|---|---:|---|---|---:|---:|---:|
| Bias delta, induction vs full | -0.0285 | [-0.0400, -0.0172] | Wilcoxon | 1.54e-07 | 4.23e-07 | -0.51 |
| Bias delta, induction vs random | -0.0255 | [-0.0370, -0.0145] | Wilcoxon | 8.08e-07 | 1.08e-06 | -0.46 |
| Entropy, induction vs full | -0.0417 | [-0.0552, -0.0274] | Wilcoxon | 2.11e-07 | 4.23e-07 | -0.59 |
| Entropy, induction vs random | -0.0289 | [-0.0430, -0.0142] | Wilcoxon | 2.21e-05 | 2.21e-05 | -0.40 |

Interpretation:
- Ablating `L5H5` and `L5H1` significantly reduced repeated-token bias.
- The same intervention also significantly reduced entropy over the candidate set, so it did not make the model more diffuse overall.
- Utility degraded more than the random-head control, suggesting the intervention was not a clean leakage-specific fix.

### Sensitivity Runs
I ran two smaller sweeps to test whether the conclusion depended on which high-scoring heads were chosen.

| Run | Heads | Copy acc full | Copy acc ablated | Bias delta full | Bias delta ablated | Entropy full | Entropy ablated | WikiText ppl full | WikiText ppl ablated |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `logs/sweep_top2.json` | `L5H5,L6H9` | 0.542 | 0.250 | 0.0371 | 0.0804 | 0.8701 | 0.8720 | 84.47 | 86.83 |
| `logs/sweep_top3.json` | `L5H5,L6H9,L5H1` | 0.542 | 0.333 | 0.0371 | 0.0638 | 0.8701 | 0.7909 | 84.47 | 201.45 |
| `logs/final_run.log` | `L5H5,L5H1` | 0.729 | 0.823 | 0.0496 | 0.0211 | 0.8602 | 0.8186 | 89.90 | 169.74 |

The sign flip is the most important empirical result in the project. Depending on which induction-like subset is ablated, repeated-token bias either rises or falls.

## 5. Analysis & Discussion
### What the Results Show
Three findings were stable:
1. GPT-2 small contains a small number of clearly induction-like heads, clustered at the canonical mid-layer locations.
2. Those heads matter for behavior.
3. Their influence is heterogeneous rather than uniformly “leak-inducing”.

The final run alone could be read as partial support for the user hypothesis, because repeated-token bias fell after ablating two high-scoring heads. But that interpretation does not survive the sensitivity runs. When `L6H9` is included, the effect reverses: copy accuracy drops sharply and repeated-token bias rises. That means the phenomenon is not explained by “induction heads” as one monolithic class.

### Comparison to Prior Work
- The head identities match the induction-head literature, which increases confidence that the detection method is sensible.
- The instability across subsets fits the memorization literature’s warning that leakage is not likely to reduce to one tiny isolated mechanism.
- The entropy results echo the diffuse-distribution literature: making the model less likely to repeat a specific distractor does not necessarily make it more random overall.

### Error Analysis and Failure Modes
- The random-choice prompts are synthetic and evaluated on a base model, not an instruction-tuned model. This keeps the causal setup controlled, but it makes “follow the instruction to be random” a weaker behavior target.
- The copy-task metric was sensitive to exactly which heads were ablated, even among the top-scoring subset.
- Entropy and utility often moved together: ablations that reduced one kind of repeated-token bias also degraded general language modeling.

### Answer to the Research Question
The evidence here does not support the strong claim that induction heads are a large singular source of leakage. A narrower statement is better supported: some induction-like heads can modulate repeated-token bias, but the effect depends heavily on which heads are targeted and is entangled with general model quality.

## 6. Limitations
- Only one model family was tested: `gpt2-small`.
- The final run had to use moderate sample sizes because CPU-only TransformerLens hooks were slow after CUDA failed.
- The “randomness” task was synthetic rather than a benchmark from an instruction-tuned model.
- No direct training-data extraction attack was run; this study focused on prompt-induced copying pressure, not full memorization extraction.
- Wikitext utility was measured on a small subset for cost reasons.

## 7. Conclusions & Next Steps
This project found real induction-like heads in GPT-2 small, but did not find clean evidence that they are a single large source of leakage. The strongest result is actually the sensitivity: ablations over different high-scoring induction-head subsets can push repeated-token bias in opposite directions.

The next experiments I would run are:
1. Repeat the same protocol on an instruction-tuned model where “choose randomly” is behaviorally meaningful.
2. Replace hard zero ablation with softer descaling, following the repetition-curse paper, to separate circuit-specific effects from gross utility loss.
3. Combine this setup with an extraction benchmark so the same intervention is tested on both random-output prompts and actual memorization-style attacks.

## References
- Olsson et al. 2022. *In-context Learning and Induction Heads*.
- Crosbie and Shutova 2024. *Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning*.
- Ren et al. 2024. *Identifying Semantic Induction Heads to Understand In-Context Learning*.
- Wang et al. 2025. *Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models*.
- Zhang et al. 2024. *Forcing Diffuse Distributions out of Language Models*.
- Lee et al. 2021. *Deduplicating Training Data Makes Language Models Better*.
- Yu et al. 2023. *Bag of Tricks for Training Data Extraction from Language Models*.
- Huang et al. 2024. *Demystifying Verbatim Memorization in Large Language Models*.
