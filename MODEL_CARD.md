---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
library_name: transformers
pipeline_tag: text-classification
tags:
  - decision-model
  - candidate-ranking
  - qwen3.5
  - mind2web
---

# JevForge-0.8B

JevForge-0.8B is a candidate-scoring checkpoint for structured web decisions.
It fine-tunes the pretrained Qwen3.5-0.8B backbone with a shared scalar decision
head and returns a complete probability distribution over supplied candidates
without autoregressive decoding.

The checkpoint supports the Jev-style `choice`, binary `noul`, and ordered
`score` contracts used by the JevForge training, evaluation, and serving stack.

JevForge is a personal, independent research project exploring open
structured-decision models and Jev-style interfaces.

## Evaluation

The recorded evaluation uses fixed website-disjoint splits built from the same
data recipe used for the 0.6B run.

| Metric | Test | OOD |
|---|---:|---:|
| Choice top-1 in positive set | 0.579 | 0.637 |
| Exact gold choice | 0.568 | 0.619 |
| Noul accuracy | 0.826 | 0.860 |
| Noul Brier | 0.128 | 0.117 |
| Score MAE | 0.367 | 0.390 |

The test and OOD splits contain 2,400 and 1,158 questions respectively. See the
source repository for the benchmark artifact and evaluation implementation.

On the first 200 records of each frozen split, the raw Qwen3.5-0.8B backbone
used as a zero-shot JSON candidate scorer reaches 0.235 test and 0.340 OOD
choice top-1, versus 0.505 and 0.630 for JevForge-0.8B on the same slices.

## Local live demo

From the JevForge repository on an Apple Silicon Mac:

```bash
bash scripts/run_mac_demo.sh
```

The command downloads this checkpoint, selects Apple MPS automatically, starts
the Jev-compatible local endpoint, and opens an interactive page that displays
the measured inference latency, candidate probabilities, and resulting click.

## Training data

The run uses transformed records derived from
[Mind2Web](https://github.com/OSU-NLP-Group/Mind2Web) annotations sourced
through `LangAGI-Lab/Mind2Web-axtree-cleaned-lite`, with decision-label data
augmentation for fields not present in the source annotations.

Mind2Web is distributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The checkpoint does
not bundle raw or unzipped official Mind2Web test files.

## Data lineage

- Backbone: pretrained `Qwen/Qwen3.5-0.8B` weights.
- Upstream dataset: `OSU-NLP-Group/Mind2Web`.
- Processed source: `LangAGI-Lab/Mind2Web-axtree-cleaned-lite`.
- Split policy: website-disjoint train, development, calibration, test, and OOD.
- Calibration temperature: 0.9717.

## Limitations

Candidate sets are supplied to the model; candidate discovery is handled by the
calling application. Reported results cover the frozen candidate-scoring
protocol and should not be read as a general browser-agent leaderboard result.
