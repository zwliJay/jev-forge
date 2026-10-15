# Mixed Open-Jev / Mind2Web / general training (8k context)

JevForge-0.8B mixed-training recipe: same architecture, mixed
public decision data, 8192-token candidate paths, up to 16k training decisions,
with sample count selected from measured runtime, one epoch initially. This document makes no claims
about measured results or superiority.

## Data build (jevforge/build_mixed.py)

Productizes the working builders (`work/build_candidate_pools_8k.py`,
`work/build_hybrid_mix_8kctx.py`) into one CLI. The recorded run used the
archived two-stage scripts plus deterministic final subsampling; this CLI is a
reusable implementation, not a claim of byte-identical run reproduction. Sources:

* **Bekko** `hotchpotch/bekko-system-one-dataset-v0` at an exact pinned
  40-hex revision, local snapshot (`training-manifest.json`, `sources.json`,
  Parquet). Only manifest **train**-role entries with row-level
  `split == "train"` are read; evaluation/validation/test entries are never
  opened for records, selection or calibration.
* **Mind2Web** release `AndeyTait/JevForge-Mind2Web` at an exact pinned
  revision, local download with `data/train.jsonl`, `data/dev.jsonl`,
  `data/calibration.jsonl`, `SOURCE.json` and optional `checksums.sha256`
  (verified when present). Source split roles map to the same output roles.

```bash
python -m jevforge.build_mixed \
  --bekko-dir      /path/to/bekko-snapshot \
  --bekko-revision 9c673a09c31e145684ab9915871afa72ba40a383 \
  --mind2web-dir   /path/to/mind2web-public \
  --mind2web-revision 8c6fa6c9804f5b18b38a5e4dc343d37b4ed7cf2b \
  --configs-json   /path/to/selected-configs.json \
  --tokenizer      /path/to/local-qwen-tokenizer \
  --output-dir     /path/to/mixed-8k \
  --train-decisions 16000 --dev-decisions 300 --calibration-decisions 150 \
  --max-length 8192 --max-candidates 32 --question-token-budget 16384 \
  --seed 42 \
  --exclude-states /path/to/exclude-eval-states.jsonl \
  --cap-openjev 1200 --cap-general 80
```

### Semantics enforced by the builder

* **Mixture quotas are exact** (largest-remainder rounding of 60% Open-Jev /
  25% Mind2Web / 15% other Bekko): 16000 → 9600/4000/2400. Sampling is
  round-robin per task and per source config with seed-fixed shuffling;
  quota shortages fail loudly with per-task pool sizes instead of shrinking
  silently. `--openjev-oversample N` (train split only) may repeat Open-Jev
  picks to reach a quota when pools run short; copies are counted in the
  manifest. `--cap-openjev/--cap-general/--cap-configs` control per-config
  acceptance caps, and `--scan-rows` caps how many rows are read per config
  (deterministic row-group sampling), so the 2M-row snapshot is never fully
  scanned or tokenized.
* **Nothing is invented, relabelled, renormalised or truncated.** Soft
  targets are carried through candidate ids verbatim; choice criteria render
  as anonymous `option_i` in stored order (matching the S1MB anonymous-choice
  rendering); noul maps by semantic `false`/`true` regardless of stored
  order; score accepts only exact ascending `0..K-1` numeric scales — the
  native index contract — and rejects any other numeric scale whole, with an
  explicit `rejected:nonindex_score_scale` counter disclosed in the manifest.
* **Length guards reject complete questions, never truncate**: the native
  `encode_examples` guard enforces `--max-length 8192` per candidate path and
  `--question-token-budget 16384` caps the padded whole question
  (64-token-padded width × candidate count).
* **Split hygiene**: Bekko train rows are partitioned 80/10/10 by a stable
  hash of the source `group_id`; Mind2Web roles pass through unchanged. A
  canonical state never appears in two output splits (`cross_split_state`
  drops), full inputs are globally deduplicated, and the final outputs
  assert group- and state-disjointness across train/dev/calibration.
* **Evaluation exclusion** (`--exclude-states`): a JSONL of canonical or raw
  **state values only** (never labels), matched via key-order-independent,
  whitespace-normalised canonical JSON hashing. Bare values and
  `{"state": ..., "provenance": {...}}` wrappers are accepted; the file's
  SHA256, hash count and declared provenance are recorded.
* **Provenance**: `manifest.json` records the recipe id, seed, both source
  revisions, SHA256 of every input file read, per-config acceptance vs cap,
  per-split decision/task/source counts, output SHA256s, and every rejection
  counter. Upstream terms from Bekko `sources.json` and the Mind2Web
  `SOURCE.json` are retained verbatim — **no blanket relicensing** — and the
  manifest states that selected configs and their license fitness remain the
  caller's responsibility.

## Training run (unchanged trainer)

```bash
python -m jevforge.train \
  --records /path/to/mixed-8k \
  --output-dir runs/mixed-8k \
  --base-model /path/to/local-qwen3.5-0.8b \
  --backbone-init pretrained \
  --max-length 8192 \
  --steps <one epoch; ~16000 questions / batches> \
  --head-steps 40 --eval-every 100 --seed 42 \
  --questions-per-batch 8 --microbatch-tokens 32768 \
  --per-type-temperature
```

* Architecture unchanged: Qwen3.5-0.8B backbone, shared MLP scalar head, one
  softmax per question over its complete candidate set. `--max-length 8192`
  must match the builder. A single question can cost up to the 16384-token
  budget, so `--microbatch-tokens 32768` keeps whole questions in a
  microbatch; the trainer never splits candidate sets.
* One epoch initially: set `--steps` to one pass over the built schedule
  (~16000 questions at ~8 per batch ≈ 2000 steps) and bound it by the
  ~40-minute wall clock; keep `--eval-every` frequent enough for a few dev
  checkpoints. Checkpoint selection stays on dev CE; the temperature is
  fitted on the calibration split afterwards.
* Separation of concerns: train/dev/calibration above come only from Bekko
  train-role rows and the Mind2Web train/dev/calibration roles. S1MB test
  data is never an input to any of them; its state values enter only through
  the exclusion file. Independent S1MB evaluation runs against the exported
  checkpoint by Codex.

## Release plan

Codex replaces the weights in the existing HF repo `AndeyTait/JevForge-0.8B`
(replacement preserves repo history) and prepares benchmark files for the user to submit a PR. The
manifest and this recipe ship with the release; no measured scores are
claimed here.

## Recorded run configuration

The 2026-10-16 run uses 8,000 training decisions (4,800 Open-Jev,
2,000 Mind2Web, 1,200 general), 300 dev and 150 calibration decisions.
The context limit is 8,192; accepted training candidate paths actually range
up to 1,623 tokens. An 8K limit does not imply an 8K-long training curriculum.
The checkpoint schedule has 870 microbatches for one full epoch, with 12
head-only warmup steps, 32 questions per batch, a 24,000 padded-token
microbatch budget and gradient checkpointing. Best checkpoint selection
uses dev CE at steps 290, 580 and 870; calibration is independent.

The recorded runtime pins Torch 2.8.0+cu128, Transformers 5.19.0,
flash-linear-attention 0.5.2 and causal-conv1d 1.7.0. To use these explicit
CUDA bindings, call `jevforge.runtime_kernels.enable_native_kernels()`
before loading the backbone, then invoke `jevforge.train.main()`. The helper
fails on other Transformers versions rather than silently guessing an API.
BF16, TF32, fused AdamW, native CUDA convolution, FLA recurrence and gradient
checkpointing are enabled. `torch.compile` and Hub kernel replacement are
not used. W&B project is `760243703-renmin-university-of-china/jevforge`,
run `hybrid8kctx20261016`, saved offline at the user's request.

The released manifest, actual assembly scripts, launcher, independent source
audit and metrics are the run evidence. Benchmark evaluation uses the same
CUDA bindings with the official S1MB JevForge adapter; this override and
context settings must accompany submissions. The user submits the PR.

## Completed evaluation

The full 137-benchmark English evaluation is complete with zero failures and
zero abstentions. See [the fixed-configuration report](S1MB_MIXED_20261016.md).
The user receives validated files and the PR body; no leaderboard PR has been
submitted by this workflow.
