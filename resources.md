# Resources Catalog

## Summary

This catalog covers the papers, datasets, and code gathered for studying whether induction heads are a major source of leakage or pathological copying in language models.

## Papers

Total papers downloaded: 10

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Deduplicating Training Data Makes Language Models Better | Lee et al. | 2021 | `papers/2107.06499_deduplicating_training_data_makes_lms_better.pdf` | deduplication reduces memorization and is a major control variable |
| In-context Learning and Induction Heads | Olsson et al. | 2022 | `papers/2209.11895_in_context_learning_and_induction_heads.pdf` | foundational induction-head mechanism paper |
| Preventing Verbatim Memorization in Language Models Gives a False Sense of Privacy | Ippolito et al. | 2022 | `papers/2210.17546_preventing_verbatim_memorization_false_privacy.pdf` | exact-copy prevention does not remove leakage |
| Bag of Tricks for Training Data Extraction from Language Models | Yu et al. | 2023 | `papers/2302.04460_bag_of_tricks_training_data_extraction.pdf` | extraction baseline and attack tuning |
| Training Data Extraction From Pre-trained Language Models: A Survey | Ishihara | 2023 | `papers/2305.16157_training_data_extraction_survey.pdf` | memorization and leakage taxonomy |
| Identifying Semantic Induction Heads to Understand In-Context Learning | Ren et al. | 2024 | `papers/2402.13055_identifying_semantic_induction_heads.pdf` | semantic/relational induction heads |
| Forcing Diffuse Distributions out of Language Models | Zhang et al. | 2024 | `papers/2404.10859_forcing_diffuse_distributions.pdf` | randomness and entropy failure benchmark plus mitigation |
| Induction Heads as an Essential Mechanism for Pattern Matching in In-context Learning | Crosbie and Shutova | 2024 | `papers/2407.07011_induction_heads_pattern_matching_icl.pdf` | causal ablations in Llama-3-8B and InternLM2-20B |
| Demystifying Verbatim Memorization in Large Language Models | Huang et al. | 2024 | `papers/2407.17817_demystifying_verbatim_memorization.pdf` | controlled memorization is distributed, not obviously one circuit |
| Induction Head Toxicity Mechanistically Explains Repetition Curse in Large Language Models | Wang et al. | 2025 | `papers/2505.13514_induction_head_toxicity_repetition_curse.pdf` | direct repetition/induction-head link; stored from arXiv `v1` |

See `papers/README.md` for more context.

## Datasets

Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-103 | Hugging Face `Salesforce/wikitext` | 1,801,350 train rows, about 548 MB | natural language modeling | `datasets/wikitext_103_v1/` | natural-text baseline |
| Synthetic Copy Task | Hugging Face `flaitenberger/synthetic_copy_task` | 900,000 train / 100,000 test, about 3.55 GB | controlled copy/retrieval | `datasets/synthetic_copy_task/` | best direct probe of induction-style copying |
| Needle-in-a-Haystack Retrieval | Hugging Face `dwzhu/needle_in_a_haystack_retrieval` | 400 queries / 800 docs, about 28.5 MB | long-context retrieval | `datasets/needle_in_a_haystack_retrieval/` | raw JSON benchmark asset |

See `datasets/README.md` for download and loading instructions.

## Code Repositories

Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens | attention/head analysis and patching | `code/TransformerLens/` | main mechanism-analysis toolkit |
| nanoGPT | https://github.com/karpathy/nanoGPT | train small GPTs cheaply | `code/nanoGPT/` | useful for toy reproductions |
| deduplicate-text-datasets | https://github.com/google-research/deduplicate-text-datasets | corpus deduplication | `code/deduplicate-text-datasets/` | controls a major leakage confound |
| LM-Extraction | https://github.com/weichen-yu/LM-Extraction | extraction attack baselines | `code/LM-Extraction/` | depends on extraction benchmark data |
| lm-extraction-benchmark | https://github.com/google-research/lm-extraction-benchmark | targeted extraction benchmark | `code/lm-extraction-benchmark/` | includes format, pointers, and simple baseline |

See `code/README.md` for key files.

## Resource Gathering Notes

### Search Strategy

Manual arXiv, Hugging Face, and GitHub search was used. The local `paper-finder` helper did not return usable results because the expected local service endpoint was unavailable, so paper selection was done by query expansion around three themes: induction heads, leakage/memorization, and random/diffuse generation.

### Selection Criteria

- Direct relevance to induction heads or copying circuits.
- Direct relevance to leakage, memorization, or extraction evaluation.
- At least one paper covering the random-output symptom.
- Preference for resources with reusable code or benchmarks.

### Challenges Encountered

- `uv add` initially failed because the workspace had no importable package; a minimal `research_workspace` package fixed isolated dependency management.
- arXiv paper `2505.13514` has a withdrawn current version with no PDF; the saved PDF comes from `v1`.
- `dwzhu/needle_in_a_haystack_retrieval` is easiest to use as raw JSON rather than through the current `datasets` loader.

### Gaps and Workarounds

- No single paper directly proves induction heads are a large source of privacy leakage.
- Repetition-curse evidence was used as the closest mechanism-level support.
- Leakage evaluation resources were added separately through extraction papers and code repos.

## Recommendations for Experiment Design

1. Primary dataset(s): use `synthetic_copy_task` for causal induction-head interventions, then validate on `wikitext_103_v1` for natural text behavior.
2. Baseline methods: full model, random-head ablation, induction-head ablation, induction-head descaling, and extraction baselines from `LM-Extraction`.
3. Evaluation metrics: copy-task accuracy, extraction recall at fixed error budget, exact/fuzzy memorization rate, output entropy on random-choice prompts, and downstream utility retention.
4. Code to adapt/reuse: `TransformerLens` for head identification and intervention, `nanoGPT` for cheap toy-model training, `deduplicate-text-datasets` for confound control, and `lm-extraction-benchmark` plus `LM-Extraction` for leakage evaluation.
