from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from scipy import stats
from transformer_lens import HookedTransformer, utils
from transformer_lens.head_detector import detect_head


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
LOGS_DIR = ROOT / "logs"


@dataclass
class Config:
    model_name: str = "gpt2-small"
    seed: int = 42
    n_head_detection_prompts: int = 96
    n_copy_prompts: int = 256
    n_random_prompts: int = 240
    n_wikitext_samples: int = 24
    top_k_heads: int = 2
    n_random_head_sets: int = 5
    max_wikitext_tokens: int = 64
    alpha: float = 0.05


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_cmd(cmd: list[str]) -> str:
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        return (completed.stdout or completed.stderr).strip()
    except OSError as exc:
        return f"command_failed: {exc}"


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    if len(values) == 0:
        return float("nan"), float("nan")
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    if diff.std(ddof=1) == 0:
        return 0.0
    return diff.mean() / diff.std(ddof=1)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict[str, float | str]:
    diff = a - b
    if len(diff) < 3:
        return {"test": "insufficient_n", "statistic": float("nan"), "p_value": float("nan")}
    shapiro_p = stats.shapiro(diff).pvalue if len(diff) <= 5000 else 1.0
    if shapiro_p > 0.05:
        stat, p_value = stats.ttest_rel(a, b)
        test_name = "paired_t"
    else:
        stat, p_value = stats.wilcoxon(a, b, alternative="two-sided")
        test_name = "wilcoxon"
    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "normality_p": float(shapiro_p),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    ranked = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    n = len(p_values)
    running_min = 1.0
    for rank in range(n, 0, -1):
        index, p_value = ranked[rank - 1]
        adjusted_value = min(running_min, p_value * n / rank)
        adjusted[index] = adjusted_value
        running_min = adjusted_value
    return adjusted


def detect_gpu_status() -> dict[str, object]:
    nvidia_output = run_cmd(
        ["bash", "-lc", "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null || echo NO_GPU"]
    )
    cuda_warning = None
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # pragma: no cover - defensive
        cuda_available = False
        cuda_warning = str(exc)
    if not cuda_available and torch.cuda.device_count() > 0 and cuda_warning is None:
        cuda_warning = "GPU visible to nvidia-smi but unavailable to PyTorch; likely CUDA/runtime mismatch."
    return {
        "nvidia_smi": nvidia_output,
        "torch_cuda_available": cuda_available,
        "torch_device_count": int(torch.cuda.device_count()),
        "torch_cuda_warning": cuda_warning,
    }


def choose_candidates(model: HookedTransformer) -> list[str]:
    preferred = [
        " red",
        " blue",
        " green",
        " yellow",
        " black",
        " white",
        " orange",
        " purple",
        " silver",
        " brown",
        " gold",
        " pink",
    ]
    single_token = []
    for candidate in preferred:
        token_ids = model.to_tokens(candidate, prepend_bos=False).squeeze(0)
        if token_ids.numel() == 1:
            single_token.append(candidate)
    if len(single_token) < 6:
        raise RuntimeError(f"Need at least 6 single-token candidates, found {single_token}")
    return single_token[:6]


def token_id(model: HookedTransformer, text: str) -> int:
    encoded = model.to_tokens(text, prepend_bos=False).squeeze(0)
    assert encoded.numel() == 1, f"{text!r} is not a single token"
    return int(encoded.item())


def build_induction_prompt(sequence: list[str]) -> tuple[str, str]:
    prompt = "".join(sequence + sequence[:-1])
    target = sequence[-1]
    return prompt, target


def make_head_detection_prompts(candidates: list[str], n_prompts: int, rng: random.Random) -> list[str]:
    prompts = []
    for _ in range(n_prompts):
        sequence = rng.sample(candidates, k=4)
        prompt, _ = build_induction_prompt(sequence)
        prompts.append(prompt)
    return prompts


def score_heads(model: HookedTransformer, prompts: list[str]) -> np.ndarray:
    scores = []
    for prompt in prompts:
        tensor = detect_head(model, prompt, "induction_head", exclude_bos=True)
        scores.append(tensor.cpu().numpy())
    return np.mean(np.stack(scores), axis=0)


def top_heads_from_scores(scores: np.ndarray, k: int) -> list[tuple[int, int]]:
    flat = []
    for layer in range(scores.shape[0]):
        for head in range(scores.shape[1]):
            flat.append(((layer, head), float(scores[layer, head])))
    flat.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in flat[:k]]


def build_random_head_sets(
    n_layers: int,
    n_heads: int,
    excluded: set[tuple[int, int]],
    k: int,
    n_sets: int,
    seed: int,
) -> list[list[tuple[int, int]]]:
    universe = [(layer, head) for layer in range(n_layers) for head in range(n_heads) if (layer, head) not in excluded]
    rng = random.Random(seed)
    return [rng.sample(universe, k=k) for _ in range(n_sets)]


def zero_head_output(z: torch.Tensor, hook, heads_to_zero: list[int]) -> torch.Tensor:
    if not heads_to_zero:
        return z
    z = z.clone()
    z[:, :, heads_to_zero, :] = 0
    return z


def hooks_for_head_set(heads: list[tuple[int, int]]) -> list[tuple[str, Callable]]:
    layer_to_heads: dict[int, list[int]] = {}
    for layer, head in heads:
        layer_to_heads.setdefault(layer, []).append(head)
    hooks = []
    for layer, layer_heads in layer_to_heads.items():
        hooks.append(
            (
                utils.get_act_name("z", layer, "attn"),
                partial(zero_head_output, heads_to_zero=sorted(layer_heads)),
            )
        )
    return hooks


def model_logits(
    model: HookedTransformer,
    prompt: str,
    hooks: list[tuple[str, Callable]] | None = None,
) -> torch.Tensor:
    tokens = model.to_tokens(prompt)
    if hooks:
        logits = model.run_with_hooks(tokens, return_type="logits", fwd_hooks=hooks)
    else:
        logits = model(tokens, return_type="logits")
    return logits[0, -1, :].detach().cpu()


def evaluate_copy_task(
    model: HookedTransformer,
    candidates: list[str],
    induction_heads: list[tuple[int, int]],
    random_head_sets: list[list[tuple[int, int]]],
    n_prompts: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    rows = []
    prompt_rows = []
    condition_to_heads: dict[str, list[tuple[int, int]]] = {"full": []}
    condition_to_heads["induction_ablation"] = induction_heads
    for index, head_set in enumerate(random_head_sets):
        condition_to_heads[f"random_ablation_{index}"] = head_set

    for prompt_index in range(n_prompts):
        sequence = rng.sample(candidates, k=4)
        prompt, target = build_induction_prompt(sequence)
        target_id = token_id(model, target)
        for condition, heads in condition_to_heads.items():
            hooks = hooks_for_head_set(heads) if heads else None
            logits = model_logits(model, prompt, hooks=hooks)
            prediction_id = int(torch.argmax(logits).item())
            prompt_rows.append(
                {
                    "prompt_index": prompt_index,
                    "condition": condition,
                    "prompt": prompt,
                    "target": target.strip(),
                    "prediction": model.to_string(torch.tensor([prediction_id])),
                    "correct": int(prediction_id == target_id),
                    "target_logit": float(logits[target_id]),
                }
            )

    prompt_df = pd.DataFrame(prompt_rows)
    for condition, group in prompt_df.groupby("condition"):
        rows.append(
            {
                "condition": condition,
                "copy_accuracy": float(group["correct"].mean()),
                "mean_target_logit": float(group["target_logit"].mean()),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)
    return summary_df, prompt_df


RANDOM_TEMPLATES = [
    "Possible answers:{options}. Previous answers:{repeated},{repeated},{repeated}. Choose one answer with no preference. Answer:",
    "Options:{options}. Seen earlier:{repeated},{repeated},{repeated}. Pick one option at random. Choice:",
    "Valid labels:{options}. Earlier labels ended with {repeated},{repeated},{repeated}. Ignore the history and output one valid label. Label:",
]


def candidate_distribution(logits: torch.Tensor, candidate_ids: list[int]) -> np.ndarray:
    subset = logits[candidate_ids]
    probs = F.softmax(subset, dim=0).numpy()
    return probs


def normalized_entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy / np.log(len(probs)))


def kl_to_uniform(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    uniform = np.full_like(probs, 1.0 / len(probs))
    return float(np.sum(probs * (np.log(probs) - np.log(uniform))))


def evaluate_random_bias(
    model: HookedTransformer,
    candidates: list[str],
    induction_heads: list[tuple[int, int]],
    random_head_sets: list[list[tuple[int, int]]],
    n_prompts: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    condition_to_heads: dict[str, list[tuple[int, int]]] = {"full": [], "induction_ablation": induction_heads}
    for index, head_set in enumerate(random_head_sets):
        condition_to_heads[f"random_ablation_{index}"] = head_set

    rows = []
    for prompt_index in range(n_prompts):
        option_order = rng.sample(candidates, k=len(candidates))
        option_ids = [token_id(model, candidate) for candidate in option_order]
        repeated = rng.choice(option_order)
        repeated_index = option_order.index(repeated)
        template = RANDOM_TEMPLATES[prompt_index % len(RANDOM_TEMPLATES)]
        options_text = ",".join(option_order)
        neutral_prompt = f"Possible answers:{options_text}. Choose one answer with no preference. Answer:"
        biased_prompt = template.format(options=options_text, repeated=repeated)

        for condition, heads in condition_to_heads.items():
            hooks = hooks_for_head_set(heads) if heads else None
            neutral_logits = model_logits(model, neutral_prompt, hooks=hooks)
            biased_logits = model_logits(model, biased_prompt, hooks=hooks)
            neutral_probs = candidate_distribution(neutral_logits, option_ids)
            biased_probs = candidate_distribution(biased_logits, option_ids)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "condition": condition,
                    "template_id": prompt_index % len(RANDOM_TEMPLATES),
                    "repeated_token": repeated.strip(),
                    "repeated_index": repeated_index,
                    "neutral_repeated_prob": float(neutral_probs[repeated_index]),
                    "biased_repeated_prob": float(biased_probs[repeated_index]),
                    "bias_delta": float(biased_probs[repeated_index] - neutral_probs[repeated_index]),
                    "normalized_entropy": normalized_entropy(biased_probs),
                    "kl_to_uniform": kl_to_uniform(biased_probs),
                    "repeated_is_top1": int(np.argmax(biased_probs) == repeated_index),
                    "top1_token": option_order[int(np.argmax(biased_probs))].strip(),
                }
            )
    return pd.DataFrame(rows)


def mean_random_condition(random_bias_df: pd.DataFrame) -> pd.DataFrame:
    random_df = random_bias_df[random_bias_df["condition"].str.startswith("random_ablation_")].copy()
    if random_df.empty:
        raise RuntimeError("Expected random ablation rows")
    metric_cols = [
        "neutral_repeated_prob",
        "biased_repeated_prob",
        "bias_delta",
        "normalized_entropy",
        "kl_to_uniform",
        "repeated_is_top1",
    ]
    aggregated = random_df.groupby("prompt_index")[metric_cols].mean().reset_index()
    aggregated["condition"] = "random_ablation_mean"
    return aggregated


def select_wikitext_samples(dataset: DatasetDict, n_samples: int, max_tokens: int) -> list[str]:
    texts = []
    for item in dataset["test"]:
        text = item["text"].strip()
        if len(text.split()) >= 10:
            texts.append(text)
        if len(texts) >= n_samples * 4:
            break
    rng = random.Random(0)
    rng.shuffle(texts)
    return texts[:n_samples]


def text_nll(model: HookedTransformer, text: str, hooks: list[tuple[str, Callable]] | None, max_tokens: int) -> float:
    tokens = model.to_tokens(text)
    if tokens.shape[1] > max_tokens:
        tokens = tokens[:, :max_tokens]
    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    if hooks:
        logits = model.run_with_hooks(inputs, return_type="logits", fwd_hooks=hooks)
    else:
        logits = model(inputs, return_type="logits")
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(-target_log_probs.mean().item())


def evaluate_wikitext_utility(
    model: HookedTransformer,
    dataset: DatasetDict,
    induction_heads: list[tuple[int, int]],
    random_head_sets: list[list[tuple[int, int]]],
    n_samples: int,
    max_tokens: int,
) -> pd.DataFrame:
    texts = select_wikitext_samples(dataset, n_samples=n_samples, max_tokens=max_tokens)
    rows = []
    condition_to_heads: dict[str, list[tuple[int, int]]] = {"full": [], "induction_ablation": induction_heads}
    condition_to_heads["random_ablation_control"] = random_head_sets[0]

    for sample_index, text in enumerate(texts):
        for condition, heads in condition_to_heads.items():
            hooks = hooks_for_head_set(heads) if heads else None
            nll = text_nll(model, text, hooks=hooks, max_tokens=max_tokens)
            rows.append(
                {
                    "sample_index": sample_index,
                    "condition": condition,
                    "nll": nll,
                    "perplexity": float(math.exp(nll)),
                }
            )
    return pd.DataFrame(rows)


def summarize_datasets() -> dict[str, object]:
    synthetic = load_from_disk(ROOT / "datasets" / "synthetic_copy_task")
    wikitext = load_from_disk(ROOT / "datasets" / "wikitext_103_v1")
    synthetic_lengths = [row["input_len"] for row in synthetic["train"].select(range(256))]
    non_empty_wikitext = [row["text"] for row in wikitext["test"] if row["text"].strip()]
    return {
        "synthetic_copy_task": {
            "splits": {name: int(len(ds)) for name, ds in synthetic.items()},
            "mean_input_len_sample": float(np.mean(synthetic_lengths)),
            "sample_text": synthetic["test"][0]["text"][:200],
        },
        "wikitext_103_v1": {
            "splits": {name: int(len(ds)) for name, ds in wikitext.items()},
            "non_empty_test_rows": int(len(non_empty_wikitext)),
            "sample_text": non_empty_wikitext[0][:200],
        },
    }


def build_head_score_figure(scores: np.ndarray, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.heatmap(scores, cmap="viridis")
    plt.title("Mean Induction Score by Head")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_copy_figure(copy_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = copy_df.copy()
    plot_df["condition"] = plot_df["condition"].replace({"random_ablation_0": "random_ablation"})
    if "random_ablation" not in set(plot_df["condition"]):
        random_rows = copy_df[copy_df["condition"].str.startswith("random_ablation_")]
        plot_df = pd.concat(
            [
                copy_df[~copy_df["condition"].str.startswith("random_ablation_")],
                pd.DataFrame(
                    [
                        {
                            "condition": "random_ablation",
                            "copy_accuracy": float(random_rows["copy_accuracy"].mean()),
                            "mean_target_logit": float(random_rows["mean_target_logit"].mean()),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    plot_df = plot_df[plot_df["condition"].isin(["full", "induction_ablation", "random_ablation"])]
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_df, x="condition", y="copy_accuracy", color="#4C72B0")
    plt.ylim(0, 1)
    plt.title("Copy Task Accuracy")
    plt.xlabel("")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_random_bias_figure(random_df: pd.DataFrame, output_path: Path) -> None:
    induction = random_df[random_df["condition"] == "induction_ablation"].copy()
    full = random_df[random_df["condition"] == "full"].copy()
    random_mean = mean_random_condition(random_df)
    merged = full[["prompt_index", "bias_delta", "normalized_entropy"]].rename(
        columns={"bias_delta": "full_bias_delta", "normalized_entropy": "full_entropy"}
    )
    merged = merged.merge(
        induction[["prompt_index", "bias_delta", "normalized_entropy"]].rename(
            columns={"bias_delta": "induction_bias_delta", "normalized_entropy": "induction_entropy"}
        ),
        on="prompt_index",
    ).merge(
        random_mean[["prompt_index", "bias_delta", "normalized_entropy"]].rename(
            columns={"bias_delta": "random_bias_delta", "normalized_entropy": "random_entropy"}
        ),
        on="prompt_index",
    )
    plot_df = pd.DataFrame(
        {
            "condition": np.repeat(["full", "induction_ablation", "random_ablation_mean"], repeats=len(merged)),
            "bias_delta": np.concatenate(
                [merged["full_bias_delta"], merged["induction_bias_delta"], merged["random_bias_delta"]]
            ),
            "normalized_entropy": np.concatenate(
                [merged["full_entropy"], merged["induction_entropy"], merged["random_entropy"]]
            ),
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(data=plot_df, x="condition", y="bias_delta", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Repeated-Token Bias Delta")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Biased - neutral probability")
    sns.boxplot(data=plot_df, x="condition", y="normalized_entropy", ax=axes[1], color="#55A868")
    axes[1].set_title("Entropy Over Candidate Set")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Normalized entropy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_wikitext_figure(wikitext_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    sns.barplot(data=wikitext_df, x="condition", y="perplexity", color="#C44E52")
    plt.title("WikiText Utility Check")
    plt.xlabel("")
    plt.ylabel("Perplexity")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def compile_summary(
    config: Config,
    scores: np.ndarray,
    induction_heads: list[tuple[int, int]],
    copy_summary_df: pd.DataFrame,
    random_bias_df: pd.DataFrame,
    wikitext_df: pd.DataFrame,
    environment: dict[str, object],
    dataset_summary: dict[str, object],
) -> dict[str, object]:
    random_mean = mean_random_condition(random_bias_df)
    full_df = random_bias_df[random_bias_df["condition"] == "full"].sort_values("prompt_index")
    induction_df = random_bias_df[random_bias_df["condition"] == "induction_ablation"].sort_values("prompt_index")
    random_mean = random_mean.sort_values("prompt_index")

    comparisons = {}
    p_values = []
    comparison_pairs = {
        "bias_delta_induction_vs_full": (induction_df["bias_delta"].to_numpy(), full_df["bias_delta"].to_numpy()),
        "bias_delta_induction_vs_random": (
            induction_df["bias_delta"].to_numpy(),
            random_mean["bias_delta"].to_numpy(),
        ),
        "entropy_induction_vs_full": (
            induction_df["normalized_entropy"].to_numpy(),
            full_df["normalized_entropy"].to_numpy(),
        ),
        "entropy_induction_vs_random": (
            induction_df["normalized_entropy"].to_numpy(),
            random_mean["normalized_entropy"].to_numpy(),
        ),
    }
    for name, (a, b) in comparison_pairs.items():
        stats_result = paired_test(a, b)
        p_values.append(float(stats_result["p_value"]))
        comparisons[name] = {
            "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)),
            "mean_diff": float(np.mean(a - b)),
            "bootstrap_ci_diff": bootstrap_ci(a - b),
            "cohens_d_paired": cohens_d_paired(a, b),
            **stats_result,
        }

    adjusted = benjamini_hochberg(p_values)
    for adjusted_value, key in zip(adjusted, comparisons):
        comparisons[key]["bh_adjusted_p"] = float(adjusted_value)

    random_rows = copy_summary_df[copy_summary_df["condition"].str.startswith("random_ablation_")]
    summary = {
        "config": asdict(config),
        "environment": environment,
        "dataset_summary": dataset_summary,
        "top_induction_heads": [{"layer": layer, "head": head, "score": float(scores[layer, head])} for layer, head in induction_heads],
        "copy_task": {
            "full_accuracy": float(copy_summary_df.loc[copy_summary_df["condition"] == "full", "copy_accuracy"].iloc[0]),
            "induction_ablation_accuracy": float(
                copy_summary_df.loc[copy_summary_df["condition"] == "induction_ablation", "copy_accuracy"].iloc[0]
            ),
            "random_ablation_accuracy_mean": float(random_rows["copy_accuracy"].mean()),
        },
        "random_task": {
            "full_bias_delta_mean": float(full_df["bias_delta"].mean()),
            "induction_bias_delta_mean": float(induction_df["bias_delta"].mean()),
            "random_bias_delta_mean": float(random_mean["bias_delta"].mean()),
            "full_entropy_mean": float(full_df["normalized_entropy"].mean()),
            "induction_entropy_mean": float(induction_df["normalized_entropy"].mean()),
            "random_entropy_mean": float(random_mean["normalized_entropy"].mean()),
            "comparisons": comparisons,
        },
        "wikitext_utility": {
            condition: {
                "mean_nll": float(group["nll"].mean()),
                "mean_perplexity": float(group["perplexity"].mean()),
            }
            for condition, group in wikitext_df.groupby("condition")
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-prompts", type=int, default=96)
    parser.add_argument("--copy-prompts", type=int, default=256)
    parser.add_argument("--random-prompts", type=int, default=240)
    parser.add_argument("--wikitext-samples", type=int, default=24)
    parser.add_argument("--top-k-heads", type=int, default=2)
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        n_head_detection_prompts=args.head_prompts,
        n_copy_prompts=args.copy_prompts,
        n_random_prompts=args.random_prompts,
        n_wikitext_samples=args.wikitext_samples,
        top_k_heads=args.top_k_heads,
    )
    set_seed(config.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    start_time = time.time()
    environment = detect_gpu_status()
    environment["python"] = sys.version
    environment["torch"] = torch.__version__

    dataset_summary = summarize_datasets()
    synthetic_dataset = load_from_disk(ROOT / "datasets" / "synthetic_copy_task")
    wikitext_dataset = load_from_disk(ROOT / "datasets" / "wikitext_103_v1")
    environment["synthetic_rows"] = int(len(synthetic_dataset["train"]))
    environment["wikitext_rows"] = int(len(wikitext_dataset["train"]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(config.model_name, device=device)
    model.eval()
    candidates = choose_candidates(model)

    rng = random.Random(config.seed)
    prompts = make_head_detection_prompts(candidates, config.n_head_detection_prompts, rng)
    scores = score_heads(model, prompts)
    induction_heads = top_heads_from_scores(scores, config.top_k_heads)
    random_head_sets = build_random_head_sets(
        model.cfg.n_layers,
        model.cfg.n_heads,
        excluded=set(induction_heads),
        k=config.top_k_heads,
        n_sets=config.n_random_head_sets,
        seed=config.seed + 99,
    )

    copy_summary_df, copy_prompt_df = evaluate_copy_task(
        model=model,
        candidates=candidates,
        induction_heads=induction_heads,
        random_head_sets=random_head_sets,
        n_prompts=config.n_copy_prompts,
        seed=config.seed + 1,
    )
    random_bias_df = evaluate_random_bias(
        model=model,
        candidates=candidates,
        induction_heads=induction_heads,
        random_head_sets=random_head_sets,
        n_prompts=config.n_random_prompts,
        seed=config.seed + 2,
    )
    wikitext_df = evaluate_wikitext_utility(
        model=model,
        dataset=wikitext_dataset,
        induction_heads=induction_heads,
        random_head_sets=random_head_sets,
        n_samples=config.n_wikitext_samples,
        max_tokens=config.max_wikitext_tokens,
    )

    build_head_score_figure(scores, FIGURES_DIR / "head_scores.png")
    build_copy_figure(copy_summary_df, FIGURES_DIR / "copy_accuracy.png")
    build_random_bias_figure(random_bias_df, FIGURES_DIR / "random_bias_entropy.png")
    build_wikitext_figure(
        wikitext_df.groupby("condition", as_index=False).agg({"perplexity": "mean"}),
        FIGURES_DIR / "wikitext_perplexity.png",
    )

    summary = compile_summary(
        config=config,
        scores=scores,
        induction_heads=induction_heads,
        copy_summary_df=copy_summary_df,
        random_bias_df=random_bias_df,
        wikitext_df=wikitext_df,
        environment=environment,
        dataset_summary=dataset_summary,
    )
    summary["runtime_seconds"] = time.time() - start_time
    summary["candidates"] = [candidate.strip() for candidate in candidates]

    copy_summary_df.to_csv(RESULTS_DIR / "copy_task_summary.csv", index=False)
    copy_prompt_df.to_csv(RESULTS_DIR / "copy_task_prompt_level.csv", index=False)
    random_bias_df.to_csv(RESULTS_DIR / "random_bias_prompt_level.csv", index=False)
    wikitext_df.to_csv(RESULTS_DIR / "wikitext_utility.csv", index=False)
    with open(RESULTS_DIR / "dataset_summary.json", "w") as handle:
        json.dump(dataset_summary, handle, indent=2)
    with open(RESULTS_DIR / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
