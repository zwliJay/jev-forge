"""Create gold-preserving browser-choice views with harder distractors.

The augmenter never invents an action label. It reuses positives and negatives
from an existing JevForge record, ranks negatives by lexical overlap with the
task and valid elements, and creates deterministic candidate subsets. Only the
training split is expanded; held-out records are copied unchanged.
"""
import argparse
import json
import random
import re
from pathlib import Path

from .schema import SPLITS, dump_records, load_records


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def hard_negative_order(record, negative_ids, seed):
    state = json.loads(record["request"]["state"])
    criteria = record["request"]["questions"]["action"]["criteria"]
    positive_ids = {key for key, value in record["targets"]["action"].items()
                    if value > 0}
    query = tokens(state.get("task", ""))
    for candidate_id in positive_ids:
        query |= tokens(criteria[candidate_id])
    rng = random.Random(f"{seed}:{record['id']}")
    tie_breakers = {candidate_id: rng.random() for candidate_id in negative_ids}
    return sorted(negative_ids, key=lambda candidate_id: (
        -len(query & tokens(criteria[candidate_id])), tie_breakers[candidate_id]))


def augment_record(record, views=2, negatives_per_view=5, seed=41):
    question = record["request"]["questions"].get("action")
    target = record["targets"].get("action")
    if not question or question["type"] != "choice" or not target:
        return []
    positive_ids = [key for key, value in target.items() if value > 0]
    negative_ids = [key for key, value in target.items() if value == 0]
    if not positive_ids or not negative_ids:
        return []
    ordered = hard_negative_order(record, negative_ids, seed)
    state = json.loads(record["request"]["state"])
    elements = {element["id"]: element for element in state["elements"]}
    derived = []
    take = min(negatives_per_view, len(ordered))
    for view in range(views):
        offset = (view * max(1, take // 2)) % len(ordered)
        rotated = ordered[offset:] + ordered[:offset]
        selected = positive_ids + rotated[:take]
        if len(selected) < 2:
            continue
        selected_set = set(selected)
        criteria = {key: value for key, value in question["criteria"].items()
                    if key in selected_set}
        filtered_state = dict(state)
        filtered_state["elements"] = [elements[key] for key in criteria if key in elements]
        total = sum(target[key] for key in criteria)
        action_target = {key: target[key] / total for key in criteria}
        derived.append({
            "id": f"{record['id']}:browser-cf-{view + 1}",
            "schema_version": record.get("schema_version", "jevforge-record-v1"),
            "split": "train", "source_group": record["source_group"],
            "request": {
                "state": json.dumps(filtered_state, ensure_ascii=False, sort_keys=True),
                "questions": {"action": {**question, "criteria": criteria}},
            },
            "targets": {"action": action_target},
            "target_kinds": {"action": "counterfactual_gold_subset"},
            "gold": {"action": next(key for key in positive_ids if key in criteria)},
            "meta": {**record.get("meta", {}), "derived_from": record["id"],
                     "augmentation": "hard_negative_candidate_subset_v1",
                     "view": view + 1},
        })
    return derived


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--negatives-per-view", type=int, default=5)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    source, output = Path(args.records), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    counts, all_records = {}, []
    for split in SPLITS:
        records = load_records(source / f"{split}.jsonl")
        original_count = len(records)
        if split == "train":
            additions = [view for record in records for view in augment_record(
                record, args.views, args.negatives_per_view, args.seed)]
            records = records + additions
            counts["synthetic_train_views"] = len(additions)
        dump_records(records, output / f"{split}.jsonl")
        counts[split] = {"original": original_count, "output": len(records)}
        all_records.extend(records)
    dump_records(all_records, output / "all.jsonl")
    (output / "augmentation_manifest.json").write_text(json.dumps({
        "schema": "jevforge-browser-augmentation-v1",
        "source": str(source.resolve()), "seed": args.seed,
        "views": args.views, "negatives_per_view": args.negatives_per_view,
        "counts": counts,
        "held_out_policy": "dev/calibration/test/ood records unchanged",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()
