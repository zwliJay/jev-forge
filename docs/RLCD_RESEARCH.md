# RLCD research baseline

TypeSafe publicly describes Jev's training method as Reinforcement Learning
for Calibrated Decisions (RLCD): finite typed answers with complete,
calibrated probabilities. Its launch material also names a parallel sampler.
The reward function, sampler implementation, optimizer, data mixture, and
training recipe are not public.

JevForge therefore treats `jevforge.rlcd` as an independent, testable baseline,
not a reproduction claim. Starting from the supervised 0.8B checkpoint, it
optimizes:

```text
task utility + proper probability scoring + reference-policy KL
```

For a browser choice, all valid DOM candidates receive utility 1 and other
candidates receive 0. The calibration term is the multiclass Brier score over
the complete distribution. The KL term anchors updates to the supervised
checkpoint. `--mode exact` computes expected utility over the finite action
set; `--mode sampled` uses grouped samples and a within-group reward baseline.

Both modes retain frozen website-disjoint test and OOD splits. Runs report
cross-entropy, Brier score, expected utility, top-1 validity, entropy, and KL
against the supervised policy. This makes it possible to reject an RL update
that raises top-1 accuracy by destroying calibration or OOD behavior.
Checkpoint selection and KL diagnostics use fixed seeded subsets so repeated
reference-policy forwards do not dominate training time. Release comparisons
still run the saved checkpoint over every frozen test and OOD record.

Primary public references:

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe workflow evaluations](https://evals.typesafe.ai/)

Run on the training host:

```bash
bash scripts/rlcd_qwen35_08b.sh
```

The command writes the trained checkpoint locally, applies its calibrated
temperatures to every frozen test/OOD question, and emits the standard
JevForge comparison report. W&B mirrors training and policy diagnostics;
local prediction rows remain the source of record for the full benchmark.

See [RLCD_RESULTS.md](RLCD_RESULTS.md) for the fixed full-split comparison and
the observed trade-off between plain RL updates and browser hard-negative views.

## Browser-focused data augmentation

`jevforge.augment_browser` creates additional training-only decision views
from the existing gold action and candidate pool. Each view retains every
valid action and selects lexically confusable negatives, producing harder
candidate sets without inventing labels. Held-out records are copied without
augmentation, so improvements cannot come from test/OOD synthesis.

```bash
python -m jevforge.augment_browser \
  --records data/web_full \
  --output-dir data/web_browser_augmented \
  --views 2 \
  --negatives-per-view 5
```
