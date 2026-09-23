#!/usr/bin/env python3
"""Rebuild web/demo_data.json adding a real-Jev arm to the demo samples.

Re-derives the exact same seed-23 sample picks as make_demo_data, keeps the
existing JevForge/uniform bars, and calls the real Jev model
(~typesafe/jev-latest via a configured decisions endpoint) once per sample for its
choice distribution.
"""
import json
import os
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

DECISIONS = os.environ.get("JEVFORGE_DECISIONS_ENDPOINT", "")
JEV_MODEL = "~typesafe/jev-latest"
SPLITS = ("test", "ood")
PER_SPLIT = 8
SEED = 23


def ask_jev(state, question, api_key, retries=4):
    body = json.dumps({"model": JEV_MODEL, "state": state,
                       "questions": {"q": question}}).encode()
    last = None
    for attempt in range(retries):
        request = urllib.request.Request(DECISIONS, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                answer = json.loads(response.read().decode())["answers"]["q"]
            return answer["probabilities"]
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** attempt)
    print(f"jev call failed: {last}", file=sys.stderr)
    return None


def main():
    root = Path(__file__).resolve().parents[1]
    web = root / "web"
    old = json.loads((web / "demo_data.json").read_text(encoding="utf-8"))

    preds = {}
    for split in SPLITS:
        for line in (Path("artifacts/release_v1") / f"predictions_{split}.jsonl").read_text(
                encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["type"] == "choice":
                preds[(row["record_id"], split)] = row

    api_key = os.environ.get("JEVFORGE_API_KEY")
    if not api_key:
        raise SystemExit("export JEVFORGE_API_KEY first")

    rng = random.Random(SEED)
    samples = {}
    for split in SPLITS:
        records = [json.loads(line) for line in
                   (root / "data/web_pilot" / f"{split}.jsonl").read_text(
                       encoding="utf-8").splitlines()]
        picks = rng.sample(records, min(PER_SPLIT, len(records)))
        out = []
        for index, record in enumerate(picks):
            ids = list(record["request"]["questions"]["action"]["criteria"])
            row = preds.get((record["id"], split))
            if row is None:
                continue
            jevforge = dict(zip(ids, row["probabilities"]))
            uniform = {i: 1.0 / len(ids) for i in ids}
            old_sample = old["samples"][split][index]
            assert old_sample["website"] == record["source_group"], "pick mismatch"
            jev_raw = ask_jev(record["request"]["state"],
                              record["request"]["questions"]["action"], api_key)
            jev = {}
            if jev_raw:
                total = sum(float(v) for v in jev_raw.values()) or 1.0
                jev = {i: float(jev_raw.get(i, 0.0)) / total for i in ids}
            probabilities = {"jevforge": jevforge, "uniform": uniform}
            if jev:
                probabilities["jev"] = jev
            out.append({
                "record_id": record["id"],
                "website": record["source_group"], "split": split,
                "question_type": "choice", "task": old_sample["task"],
                "candidates": old_sample["candidates"], "gold": record["meta"]["positive_ids"],
                "probabilities": probabilities,
            })
            print(f"{split}[{index}] {record['source_group']}: jev "
                  f"{'ok' if jev else 'MISSING'}", flush=True)
        samples[split] = out

    meta = dict(old["meta"])
    meta["arms"] = {
        "jevforge": "JevForge 0.6B (local)",
        "jev": "real Jev-1.13 (hosted API)",
        "uniform": "uniform",
    }
    (web / "demo_data.json").write_text(
        json.dumps({"meta": meta, "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    counts = {s: {a: sum(1 for i in v if a in i["probabilities"])
                  for a in ("jevforge", "jev", "uniform")}
              for s, v in samples.items()}
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
