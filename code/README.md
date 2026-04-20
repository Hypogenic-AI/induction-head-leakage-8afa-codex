# Cloned Repositories

## TransformerLens
- URL: https://github.com/TransformerLensOrg/TransformerLens
- Purpose: mechanistic interpretability toolkit for loading models, caching activations, patching heads, and running attention analyses.
- Location: `code/TransformerLens/`
- Key files: `README.md`, `transformer_lens/`, `demos/`
- Notes: best starting point for identifying, ablating, and patching candidate induction heads in open models.

## nanoGPT
- URL: https://github.com/karpathy/nanoGPT
- Purpose: simple GPT training code for small-scale reproduction experiments.
- Location: `code/nanoGPT/`
- Key files: `train.py`, `model.py`, `sample.py`, `data/`
- Notes: useful for training toy models on WikiText or synthetic copy data where induction heads can be studied cheaply. The repo is marked deprecated upstream, but it remains a practical minimal baseline.

## deduplicate-text-datasets
- URL: https://github.com/google-research/deduplicate-text-datasets
- Purpose: exact and near-duplicate detection for text corpora.
- Location: `code/deduplicate-text-datasets/`
- Key files: `src/main.rs`, `scripts/load_dataset.py`, `scripts/make_suffix_array.py`
- Notes: directly relevant for controlling a major leakage confound. Use before training if the goal is to attribute copying to model circuits rather than duplicated training examples.

## LM-Extraction
- URL: https://github.com/weichen-yu/LM-Extraction
- Purpose: training-data extraction baselines from the ICML 2023 paper.
- Location: `code/LM-Extraction/`
- Key files: `baseline_ori.py`, `baseline_method.py`, `readme.md`
- Notes: provides extraction hyperparameters and a stronger baseline than naive generate-then-rank attacks.

## lm-extraction-benchmark
- URL: https://github.com/google-research/lm-extraction-benchmark
- Purpose: benchmark data format and baseline evaluation for targeted extraction attacks.
- Location: `code/lm-extraction-benchmark/`
- Key files: `baseline/simple_baseline.py`, `load_dataset.py`, `datasets/`
- Notes: complements `LM-Extraction`; needed to reproduce targeted suffix extraction evaluations. The official repo includes pointers and formats, while some raw data are hosted externally.
