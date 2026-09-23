# Dataset format: from gold actions to JevForge records

JevForge does not train directly on a browser trajectory. It converts each
annotated decision step into a self-contained record containing the current
state, typed questions, complete candidate sets, target distributions, and a
group key used to prevent train/test leakage.

Mind2Web already supplies the essential gold signal. The conversion implemented
by `jevforge/build_web.py` is:

| Mind2Web field | JevForge field | Conversion |
|---|---|---|
| `action_uid` | `id` | Stable identifier for one decision step |
| `confirmed_task` | `request.state.task` | User goal copied into the canonical state |
| `operation.original_op` | `request.state.operation` | Interaction type, such as `CLICK` |
| `pos_candidates` | `action.criteria` + `targets.action` | Gold elements receive equal probability mass |
| `neg_candidates` | `action.criteria` + `targets.action` | Sampled distractors receive zero probability |
| one sampled candidate | `is_target` | Becomes a true/false (`noul`) decision |
| `website` | `source_group` | Keeps one website in exactly one split |
| source annotation fields | `meta` | Provenance retained for inspection |

Candidate IDs such as `e1` are assigned after candidates are pooled and
shuffled. If several elements are valid, the target is uniform over all of
them; the converter does not force one arbitrary answer.

## Minimal input example

This fictional annotation has the shape consumed by the converter. It is not
copied from an upstream test split.

```json
{
  "action_uid": "settings-0001-step-2",
  "annotation_id": "settings-0001",
  "confirmed_task": "Turn on email notifications",
  "operation": "{\"original_op\": \"CLICK\"}",
  "website": "example-settings",
  "domain": "settings",
  "subdomain": "notifications",
  "pos_candidates": [
    "{\"tag\":\"button\",\"attributes\":\"{\\\"aria_label\\\":\\\"Enable email notifications\\\"}\"}"
  ],
  "neg_candidates": [
    "{\"tag\":\"button\",\"attributes\":\"{\\\"aria_label\\\":\\\"Cancel\\\"}\"}"
  ],
  "target_action_reprs": "[button] Enable email notifications"
}
```

## Resulting training record

Each dataset line is one JSON object. A runnable example is checked in at
[`examples/minimal_record.jsonl`](../examples/minimal_record.jsonl).

```json
{
  "id": "settings-0001-step-2",
  "schema_version": "jevforge-record-v1",
  "split": "train",
  "source_group": "example-settings",
  "request": {
    "state": "{\"elements\":[{\"id\":\"e1\",\"text\":\"button aria_label=Enable email notifications\"},{\"id\":\"e2\",\"text\":\"button aria_label=Cancel\"}],\"operation\":\"CLICK\",\"task\":\"Turn on email notifications\",\"website\":\"example-settings\"}",
    "questions": {
      "action": {
        "type": "choice",
        "instructions": "Which element should be used next?",
        "criteria": {"e1": "Enable email notifications", "e2": "Cancel"}
      },
      "is_target": {
        "type": "noul",
        "instructions": "The element e1 is a correct next target.",
        "criteria": {"false": "The proposition is false.", "true": "The proposition is true."}
      }
    }
  },
  "targets": {
    "action": {"e1": 1.0, "e2": 0.0},
    "is_target": {"false": 0.0, "true": 1.0}
  },
  "target_kinds": {
    "action": "uniform_over_positive_elements",
    "is_target": "deterministic_truth"
  },
  "gold": {"action": "e1", "is_target": true},
  "meta": {"annotation_id": "settings-0001", "positive_ids": ["e1"]}
}
```

The model scores one candidate path at a time, then normalizes only across
candidates belonging to the same question. Every target must therefore cover
its candidate set exactly and sum to `1.0`.

## Optional `score` data augmentation

Mind2Web provides action targets, but not every ordered score needed by later
experiments. The augmentation stage can add a question such as:

```json
"difficulty": {
  "type": "score",
  "instructions": "How difficult is this decision from the supplied state?",
  "criteria": ["easy", "medium", "hard"]
}
```

Its target uses string indices matching the ordered levels:

```json
"difficulty": {"0": 0.1, "1": 0.8, "2": 0.1}
```

Generated supervision is data augmentation, not ground truth. Keep the raw
reply cache and generation settings so it stays distinguishable from the
original gold action labels.

## Build Mind2Web-derived splits

Point the converter at locally downloaded parquet files:

```bash
python -m jevforge.build_web \
  --dataset-glob '/path/to/Mind2Web-axtree-cleaned-lite/**/*.parquet' \
  --output-dir data/web
```

It writes `train`, `dev`, `calibration`, `test`, and `ood` JSONL files plus a
manifest. Splits are deterministic and website-disjoint. Do not use a random
row split: adjacent steps from one website can otherwise leak page structure
into evaluation.

Optional augmentation and training:

```bash
export JEVFORGE_CHAT_ENDPOINT=https://your-compatible-endpoint/v1/chat/completions
export JEVFORGE_GENERATOR_MODEL=your-model-id
export JEVFORGE_API_KEY=...

python -m jevforge.synthesize \
  --records data/web/all.jsonl \
  --output-dir data/web_augmented \
  --cache artifacts/augmentation_cache.jsonl

# The release script supplies the supported training flags and evaluation steps.
bash scripts/run_web_pipeline.sh
```

Validate any record file before training:

```bash
python - <<'PY'
from jevforge.schema import load_records

records = load_records("examples/minimal_record.jsonl")
print(f"valid records: {len(records)}")
PY
```

## Training on another task

A custom dataset does not need to describe a webpage. It needs the same
decision contract:

1. **State:** a self-contained string with only the evidence available at
   decision time.
2. **Typed question:** `choice` for named alternatives, `noul` for a
   proposition, or `score` for 2–10 ordered levels.
3. **Stable candidates:** every option has a stable ID and readable description.
4. **Complete target:** a distribution over exactly those IDs; multiple valid
   answers may share probability mass.
5. **Source group:** the unit that must not cross splits, such as a website,
   application, customer, document family, or scenario template.
6. **Provenance:** source IDs, annotation version, and augmentation settings in
   `meta` or a neighboring manifest.

The same format works for tool selection, intent routing, escalation decisions,
document classification, and ordered risk scoring. Raw and derived datasets
stay local by default; redistribution remains subject to upstream terms.
