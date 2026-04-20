# Induction Head Leakage Study

This project tests whether induction heads are a major source of unintended copying on prompts that should produce diffuse or random outputs. The experiments use `gpt2-small` through TransformerLens, local synthetic and WikiText resources, and head-level ablations with random-head controls.

Key findings:
- The highest-scoring induction-like heads were consistently near the canonical GPT-2 locations `L5H5`, `L6H9`, and `L5H1`.
- The effect of ablating those heads was not stable across head subsets: some ablations increased repeated-token bias, while another reduced it but also reduced entropy and hurt WikiText perplexity.
- The strongest paired-statistics run (`top_k=2`, heads `L5H5` and `L5H1`) lowered repeated-token bias from `0.0496` to `0.0211` but also lowered normalized entropy from `0.860` to `0.819` and worsened WikiText perplexity from `89.9` to `169.7`.
- Overall, the results do not support the claim that induction heads are a single large source of leakage in this model; their role appears heterogeneous and entangled with general language-model behavior.

Reproduce:
```bash
source .venv/bin/activate
python src/run_induction_leakage_experiments.py \
  --head-prompts 48 \
  --copy-prompts 96 \
  --random-prompts 96 \
  --wikitext-samples 12 \
  --top-k-heads 2
```

Outputs:
- Main report: `REPORT.md`
- Experiment code: `src/run_induction_leakage_experiments.py`
- Saved tables: `results/`
- Figures: `figures/`
- Sensitivity runs: `logs/sweep_top2.json`, `logs/sweep_top3.json`, `logs/final_run.log`
