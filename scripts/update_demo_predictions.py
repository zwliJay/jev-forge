#!/usr/bin/env python3
"""Replace one recorded arm in a static demo with a new prediction run."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--arm", default="jevforge")
    parser.add_argument("--label", default="JevForge 0.8B")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    demo = json.loads(args.demo.read_text(encoding="utf-8"))
    updated = 0
    for split, samples in demo["samples"].items():
        records = {}
        for line in (args.records / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            state = json.loads(record["request"]["state"])
            candidate_signature = tuple(
                (candidate["id"], candidate["text"])
                for candidate in state["elements"]
            )
            key = (record["source_group"], state["task"], candidate_signature)
            if key in records:
                raise SystemExit(f"ambiguous demo record match for {key[:2]}")
            records[key] = record["id"]
        predictions = {}
        for line in (args.predictions / f"predictions_{split}.jsonl").read_text(
                encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["type"] == "choice":
                predictions[row["record_id"]] = dict(zip(row["candidate_ids"],
                                                           row["probabilities"]))
        for sample in samples:
            candidate_signature = tuple(
                (candidate["id"], candidate["text"])
                for candidate in sample["candidates"]
            )
            record_id = records.get(
                (sample["website"], sample["task"], candidate_signature)
            )
            if record_id in predictions:
                sample["probabilities"][args.arm] = predictions[record_id]
                updated += 1

    demo["meta"]["model"] = f"{args.label.lower()} + real jev-1.13, recorded"
    demo["meta"]["arms"][args.arm] = args.label
    args.output.write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"updated": updated, "output": str(args.output)}))
    if updated == 0:
        raise SystemExit("no demo samples matched the supplied records")


if __name__ == "__main__":
    main()
