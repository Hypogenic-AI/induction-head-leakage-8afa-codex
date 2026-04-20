# Downloaded Datasets

This directory contains datasets for experiments on induction heads, copying, leakage, and output diversity. Data files are excluded from git by default.

## Dataset 1: WikiText-103

### Overview
- Source: `Salesforce/wikitext` (`wikitext-103-v1`)
- Size: train 1,801,350 rows; validation 3,760; test 4,358
- Local size: about 548 MB
- Format: Hugging Face `DatasetDict`
- Task: language modeling / natural text pretraining and evaluation
- Splits: train, validation, test
- License: `cc-by-sa-3.0`

### Download Instructions

Using Hugging Face:

```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
dataset.save_to_disk("datasets/wikitext_103_v1")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext_103_v1")
```

### Sample Data

Saved at `datasets/wikitext_103_v1/samples.json`.

### Notes
- Good natural-text baseline for measuring whether induction-head interventions change copying or entropy on realistic language.
- Raw text contains many empty or formatting rows; downstream preprocessing should strip blank lines.

## Dataset 2: Synthetic Copy Task

### Overview
- Source: `flaitenberger/synthetic_copy_task`
- Size: train 900,000; test 100,000
- Local size: about 3.55 GB
- Format: Hugging Face `DatasetDict`
- Task: controlled copy/retrieval behavior
- Splits: train, test
- Features: `text`, `loss_mask`, `input_len`

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("flaitenberger/synthetic_copy_task")
dataset.save_to_disk("datasets/synthetic_copy_task")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/synthetic_copy_task")
```

### Sample Data

Saved at `datasets/synthetic_copy_task/samples.json`.

### Notes
- Best controlled dataset here for isolating induction-style copying.
- The `loss_mask` field is useful for evaluating only the copied suffix instead of the whole prompt.
- Strong candidate for ablation experiments that remove or downscale induction heads.

## Dataset 3: Needle-in-a-Haystack Retrieval Benchmark

### Overview
- Source: `dwzhu/needle_in_a_haystack_retrieval`
- Size: 400 queries, 800 corpus documents, 400 qrels
- Local size: about 28.5 MB
- Format: raw JSON benchmark file
- Task: long-context retrieval / context scanning
- Splits: single test-style file

### Download Instructions

```bash
wget https://huggingface.co/datasets/dwzhu/needle_in_a_haystack_retrieval/resolve/main/test.json \
  -O datasets/needle_in_a_haystack_retrieval/test.json
```

### Loading the Dataset

```python
import json
with open("datasets/needle_in_a_haystack_retrieval/test.json") as f:
    data = json.load(f)
```

### Sample Data

Saved at `datasets/needle_in_a_haystack_retrieval/sample_head.txt`.

### Notes
- Useful as a stress test for attention-mediated retrieval under long contexts.
- This is not a standard `datasets`-loader package in the current environment; treat it as a raw benchmark asset.

## Recommended Usage

- Primary natural corpus: `wikitext_103_v1`
- Primary controlled copying benchmark: `synthetic_copy_task`
- Auxiliary long-context retrieval probe: `needle_in_a_haystack_retrieval`
