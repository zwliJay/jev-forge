"""Build JevForge decision records from Mind2Web (axtree-cleaned-lite).

Every Mind2Web annotation step becomes one record whose request uses the
public Jev question contract:

  action    choice over the page's candidate elements
  is_target noul judging one sampled element (self-contained proposition)

States are canonical JSON strings (sorted keys). Splits are website-disjoint.
The difficulty score question is added later by the augmentation stage.
"""
import argparse
import glob
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .schema import SPLITS, dump_records

# 60/10/10/10/10 of websites by name hash, in SPLITS order.
BUCKETS = ((0, 1, 2, 3, 4, 5), (6,), (7,), (8,), (9,))


def split_for(website: str) -> str:
    bucket = int(hashlib.sha1(website.encode("utf-8")).hexdigest()[:8], 16) % 10
    for split, members in zip(SPLITS, BUCKETS):
        if bucket in members:
            return split
    raise AssertionError


def squash(value: str, limit: int) -> str:
    return " ".join((value or "").split())[:limit]


ATTRIBUTE_ORDER = ("id", "role", "aria_label", "name", "text_value", "value",
                   "placeholder", "title", "type", "alt", "label", "input_value")


def render_operation(raw) -> str:
    try:
        return squash(str(json.loads(raw).get("original_op") or raw), 24) or "ACT"
    except (ValueError, TypeError, AttributeError):
        return squash(str(raw or ""), 24) or "ACT"


def render_element(raw: str, limit: int) -> str:
    """Human-readable rendering of one Mind2Web candidate JSON blob."""
    try:
        blob = json.loads(raw)
        attrs = json.loads(blob.get("attributes") or "{}")
    except (ValueError, TypeError):
        return squash(raw, limit)
    tag = squash(str(blob.get("tag") or "node"), 20)
    bits = [f"{key}={squash(str(attrs[key]), 44)}"
            for key in ATTRIBUTE_ORDER if squash(str(attrs.get(key) or ""), 44)]
    return squash(" ".join([tag] + bits), limit) or tag


def iter_parquet(pattern):
    import pyarrow.parquet as pq

    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no parquet matches {pattern}")
    columns = ["action_uid", "operation", "pos_candidates", "neg_candidates",
               "website", "domain", "subdomain", "annotation_id",
               "confirmed_task", "target_action_reprs"]
    for path in files:
        yield from pq.read_table(path, columns=columns).to_pylist()


def make_record(row, rng, max_candidates, state_chars, desc_chars):
    positives = row.get("pos_candidates") or []
    negatives = row.get("neg_candidates") or []
    if not positives:
        return None
    room = max(0, max_candidates - len(positives))
    picked_negatives = rng.sample(negatives, min(room, len(negatives)))
    pool = [(c, True) for c in positives[:max_candidates]] + \
           [(c, False) for c in picked_negatives]
    if len(pool) < 2:
        return None
    rng.shuffle(pool)

    elements, positive_ids = [], []
    for index, (candidate, is_positive) in enumerate(pool, start=1):
        element_id = f"e{index}"
        text = render_element(candidate, desc_chars) or f"page element {element_id}"
        elements.append({"id": element_id, "text": text})
        if is_positive:
            positive_ids.append(element_id)

    header = {
        "operation": render_operation(row.get("operation")),
        "task": squash(row.get("confirmed_task") or "", 220),
        "website": squash(row.get("website") or "", 60),
    }
    state = json.dumps({"elements": elements, **header}, ensure_ascii=False,
                       sort_keys=True)
    for shrink in (48, 32, 16):  # keep records inside the char budget
        if len(state) <= state_chars:
            break
        state = json.dumps(
            {"elements": [{"id": e["id"], "text": e["text"][:shrink]} for e in elements],
             **header}, ensure_ascii=False, sort_keys=True)
    if len(state) > state_chars:
        return None

    informative = [e for e in elements if len(e["text"]) > 12] or elements
    asked = rng.choice(informative)
    asked_positive = asked["id"] in positive_ids
    website = header["website"] or "unknown"
    return {
        "id": row["action_uid"],
        "schema_version": "jevforge-record-v1",
        "split": split_for(website),
        "source_group": website,
        "request": {
            "state": state,
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "Which page element should be interacted with next to make progress on the task?",
                    "criteria": {e["id"]: e["text"] for e in elements},
                },
                "is_target": {
                    "type": "noul",
                    "instructions": f"The page element described as '{asked['text']}' is a correct next interaction target for completing the task.",
                    "criteria": {
                        "true": "The element is a correct next target for the task.",
                        "false": "The element is not a correct next target.",
                    },
                },
            },
        },
        "targets": {
            "action": {e["id"]: (1.0 / len(positive_ids) if e["id"] in positive_ids else 0.0)
                       for e in elements},
            "is_target": {"false": 0.0 if asked_positive else 1.0,
                          "true": 1.0 if asked_positive else 0.0},
        },
        "target_kinds": {"action": "uniform_over_positive_elements",
                         "is_target": "deterministic_truth"},
        "gold": {"action": sorted(positive_ids)[0], "is_target": asked_positive},
        "meta": {
            "annotation_id": row.get("annotation_id"),
            "positive_ids": positive_ids,
            "target_action_reprs": squash(row.get("target_action_reprs") or "", 160),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--state-chars", type=int, default=2600)
    parser.add_argument("--desc-chars", type=int, default=88)
    parser.add_argument("--caps", default="6000,800,400,800,800",
                        help="per-split record caps in SPLITS order")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    caps = dict(zip(SPLITS, (int(x) for x in args.caps.split(","))))
    counts, groups = Counter(), defaultdict(set)
    records, seen = [], set()
    for row in iter_parquet(args.dataset_glob):
        split = split_for(row.get("website") or "")
        if counts[split] >= caps[split] or row["action_uid"] in seen:
            continue
        record = make_record(row, rng, args.max_candidates, args.state_chars,
                             args.desc_chars)
        if record is None:
            continue
        seen.add(record["id"])
        counts[record["split"]] += 1
        groups[record["split"]].add(record["source_group"])
        records.append(record)
        if all(counts[s] >= caps[s] for s in SPLITS):
            break

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dump_records([r for r in records if r["split"] == split],
                     output / f"{split}.jsonl")
    dump_records(records, output / "all.jsonl")

    for a, b in (("train", "test"), ("train", "ood"), ("test", "ood")):
        overlap = groups[a] & groups[b]
        if overlap:
            raise SystemExit(f"website leak {a}/{b}: {sorted(overlap)}")
    manifest = {
        "source": "LangAGI-Lab/Mind2Web-axtree-cleaned-lite (parquet)",
        "seed": args.seed,
        "records": dict(counts),
        "websites": {s: len(groups[s]) for s in SPLITS},
        "max_candidates": args.max_candidates,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest["records"]))


if __name__ == "__main__":
    main()
