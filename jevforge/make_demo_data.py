"""Generate web/demo_data.json for the static side-by-side replay page.

Samples held-out test/ood records and replays the JevForge checkpoint's
recorded probabilities beside a uniform baseline.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--output", default="web/demo_data.json")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)
    preds = {}
    for split in args.splits.split(","):
        path = Path(args.predictions) / f"predictions_{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["type"] == "choice":
                preds[(row["record_id"], split)] = row
    samples = {}
    for split in args.splits.split(","):
        records = [json.loads(line) for line in
                   (Path(args.records) / f"{split}.jsonl").read_text(
                       encoding="utf-8").splitlines()]
        picks = rng.sample(records, min(args.per_split, len(records)))
        out = []
        for record in picks:
            row = preds.get((record["id"], split))
            if row is None:
                continue
            state = json.loads(record["request"]["state"])
            ids = row["candidate_ids"]
            jevforge = dict(zip(ids, row["probabilities"]))
            uniform = {i: 1.0 / len(ids) for i in ids}
            probabilities = {"jevforge": jevforge, "uniform": uniform}
            out.append({
                "website": record["source_group"], "split": split,
                "question_type": "choice",
                "task": state["task"],
                "candidates": state["elements"],
                "gold": record["meta"]["positive_ids"],
                "probabilities": probabilities,
            })
        samples[split] = out

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "meta": {"model": "jevforge-0.8b (recorded)",
                 "arms": {"jevforge": "JevForge 0.8B", "uniform": "uniform"}},
        "samples": samples,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({s: len(v) for s, v in samples.items()}))


if __name__ == "__main__":
    main()
