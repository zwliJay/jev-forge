# JevForge S1MB adapter audit — 2026-10-16

The audit found one small structured-instruction serialization mismatch. It did
not find evidence that an adapter fault explains the overall low score. An exact
old-input reproduction also recovered the earlier model's nearly-all-false behavior
through the native Predictor, without the S1MB adapter.

## Confirmed mismatch and fix

The training converter renders structured values with
`json.dumps(value, ensure_ascii=False)`, preserving source key order and default
spaces. S1MB's generic `hf_data._text` renders compact JSON with sorted keys.
The JevForge adapter reused that display string instead of the model's training
serialization. Plain strings were already identical.

An input-only comparison covered all 26,269 English decisions. Exactly **9**
question prompts differed: four in Diverse Choice and five in Diverse Score.
The remaining 26,260 native request questions were identical. Candidate IDs,
ordering, labels, weights, calibration temperatures and context limits were not
changed. Candidate descriptions were plain text in the current evaluation.

The JevForge-specific fix decodes `instructions_json` / `description_json` before
native serialization; it does not change the generic renderer for other models.
[The source patch](patches/s1mb-jevforge-native-json.patch) is based on evaluator
`a77c30cd218b06aa7f96b61682772c8aff701a87`; its local source commit is
`a0fb170cf97baed34a9381a244cb616133e0a478`. Five actual-function regression tests
pass, including plain strings, structured key order/spacing, structured criteria,
Noul order and Score level preservation. The patch has **not** been submitted or
merged upstream. An upstream source PR should accompany corrected Dataset results.

Both affected benchmarks were rerun completely, regardless of score direction,
with the same weights and inference settings. The validated combined package
retains 135 original measurements plus the two corrected measurements. It still
covers 137/137 benchmarks and 26,269 decisions, with zero failures or abstentions.
Original measurements are retained separately; this is a disclosed mixed-source
package, not a claim of one uniform fresh run.

| Baseline-adjusted task score | Original adapter | Corrected renderer |
|---|---:|---:|
| Noul | 16.46440881 | 16.46440881 |
| Choice | 20.74203885 | 20.74203885 |
| Score | 8.26140519 | 8.25391410 |
| Equal-task average | 15.15595095 | 15.15345392 |

The average changes by **−0.00249703 points**. The fix does not recover a large
missing score. No checkpoint, threshold, temperature or input policy was chosen
using evaluation scores.

## Output-boundary and runtime checks

- Actual native `Predictor.decide` and the optimized adapter were compared on
  66 synthetic questions across 22 cases. Original nonlexical Choice IDs,
  anonymous IDs, reversed Noul option order, fractional Score scales, multiple
  questions, candidate sorting and over-budget intact paths all mapped correctly.
  Maximum probability difference was below `8.5e-8`.
- On all 300 development decisions, native BF16 Predictor raw CE was
  **0.76484952**, versus recorded training CE **0.76490071**. A restored training
  forward replica gave **0.76500887**. Checkpoint loading and native evaluation
  therefore agree closely on this held-out split.
- On nine fixed development questions, optimized candidate batching versus native
  complete-question batching changed probabilities by at most 0.00703, with no
  changed argmax. Single-candidate evaluation, cache toggling and full FP32 were
  also checked; no argmax changed on these fixtures. These are sampled numerical
  checks, not a proof about every possible input.
- A separate six-question comparison called the **undecorated pure PyTorch**
  convolution/delta-rule functions and the installed CUDA implementations.
  Instrumentation recorded 18 convolution and 18 delta-rule calls on each path.
  Maximum probability difference was 0.00567, with no changed argmax. Transformers'
  default wrapper can itself call installed packages, so comparing against that
  wrapper alone is not a pure PyTorch reference check.
- All evaluated Score option scales were in ascending order. The optimized
  adapter maps probability positions back to original IDs; the evaluator uses
  original numeric values. A fractional/non-index scale alone is not an ID
  mapping error.

## Reproducing the earlier low score

The old weights were pinned to
`a7e0e71a8265ed8c4e492e4574b51afa85a85f6a`. All 100 current CREAK input hashes
matched those recorded with the old published measurements, even though the
Dataset repository/revision names differ.

| Old checkpoint on those identical inputs | Result |
|---|---|
| Published true-probability range | 0.005432–0.012848 |
| Native Predictor true-probability range | 0.005609–0.012848 |
| Native true predictions | 0 / 100 |
| Adapter true predictions | 0 / 100 |
| Maximum native/published probability difference | 0.001030 |

Thus the old CREAK failure reproduces outside the adapter. On the first 80
source Mind2Web development records, preserving original candidate IDs selected
positive targets in 58 cases; anonymizing IDs selected positives in 59. Seven
choices changed, but this fixture does not show a large anonymization-induced
collapse. These 80 records are a diagnostic slice, not a full Mind2Web score.

## What remains weak

The new model predicts true on **9.48%** of Noul decisions, while their aggregate
soft-target positive prevalence is **27.49%**. Its aggregate positive recall is
**21.81%**, versus negative recall **95.20%**. These are pooled diagnostic counts,
not the official macro-averaged leaderboard score. Training Noul positive target
mass was 31.93%, and the calibration slice was 21.43%; those proportions alone do
not establish the cause of the much stronger prediction bias.

The evidence supports investigating cross-domain discrimination and false-biased
model behavior rather than attributing the score gap to reversed labels, lost
candidates, a broken checkpoint or a large CUDA/precision discrepancy. Temperature
scaling alone cannot move a binary argmax boundary because it preserves logit order.
The audit does not establish that every possible adapter issue has been ruled out.
