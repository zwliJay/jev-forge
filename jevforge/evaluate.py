"""JevForge evaluation: correctness, calibration, and per-site breakdown.

Metrics per question type:
  choice  top-1 within positive set, exact-representative accuracy, mean
          probability mass on positives, CE
  noul    threshold accuracy, Brier, 10-bin ECE
  score   expected-level MAE, CE
Also emits a per-website table and a comparison table across runs.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def ece_bins(probs, corrects, bins=10):
    total = len(probs)
    if not total:
        return 0.0
    buckets = defaultdict(list)
    for p, c in zip(probs, corrects):
        buckets[min(bins - 1, int(p * bins))].append((p, c))
    gap = 0.0
    for entries in buckets.values():
        weight = len(entries) / total
        mean_p = math.fsum(p for p, _ in entries) / len(entries)
        mean_c = math.fsum(1.0 if c else 0.0 for _, c in entries) / len(entries)
        gap += weight * abs(mean_p - mean_c)
    return gap


def summarize(rows):
    stats = defaultdict(lambda: defaultdict(list))
    for row in rows:
        kind = row["type"]
        target = row["target"]
        ids = row["candidate_ids"]
        probs = dict(zip(ids, row["probabilities"]))
        q = [target[i] for i in ids]
        p = [probs[i] for i in ids]
        ce = -math.fsum(a * math.log(max(b, 1e-12)) for a, b in zip(q, p))
        stats[kind]["ce"].append(ce)
        if kind == "choice":
            positives = [i for i in ids if target[i] > 0]
            best = max(ids, key=probs.__getitem__)
            stats[kind]["top1_in_positives"].append(best in positives)
            stats[kind]["exact_gold"].append(best == row.get("gold"))
            stats[kind]["mass_on_positives"].append(math.fsum(probs[i] for i in positives))
        elif kind == "noul":
            p_true = probs["true"]
            correct = (p_true >= 0.5) == (target["true"] >= 0.5)
            stats[kind]["accuracy"].append(correct)
            stats[kind]["brier"].append((p_true - target["true"]) ** 2)
            stats[kind]["ece_conf"].append((max(p_true, 1.0 - p_true), correct))
        else:  # score
            expected_p = math.fsum(i * probs[str(i)] for i in range(len(ids)))
            expected_q = math.fsum(i * target[str(i)] for i in range(len(ids)))
            stats[kind]["mae"].append(abs(expected_p - expected_q))
    out = {"questions": len(rows)}
    for kind, metrics in stats.items():
        block = {}
        for name, values in metrics.items():
            if name == "ece_conf":
                block["ece"] = round(ece_bins([v[0] for v in values],
                                              [v[1] for v in values]), 4)
            else:
                block[name] = round(math.fsum(values) / len(values), 4)
        out[kind] = block
    return out


def per_group(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["type"] == "choice":
            groups[row["source_group"]].append(row)
    table = {}
    for group, items in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:20]:
        hits = 0
        for row in items:
            target = row["target"]
            ids = row["candidate_ids"]
            positives = {i for i in ids if target[i] > 0}
            best = ids[max(range(len(ids)), key=row["probabilities"].__getitem__)]
            hits += best in positives
        table[group] = {"questions": len(items), "top1_in_positives": round(hits / len(items), 4)}
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="dataset dir with <split>.jsonl")
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="run dirs containing predictions_<split>.jsonl, as name=path")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from .schema import load_records

    records_dir = Path(args.records)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"runs": {}}
    for split in args.splits.split(","):
        file = records_dir / f"{split}.jsonl"
        if not file.exists():
            continue
        meta = {}
        for record in load_records(file):
            positives = [i for i, v in record["targets"]["action"].items() if v > 0]
            meta[record["id"]] = {"source_group": record["source_group"],
                                  "positives": positives}
        rows = []
        for spec in args.predictions:
            name, _, path = spec.partition("=")
            pred_file = Path(path) / f"predictions_{split}.jsonl"
            if not pred_file.exists():
                continue
            for line in pred_file.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                info = meta.get(row["record_id"])
                if info:
                    row["source_group"] = info["source_group"]
                rows.append(row)
            report["runs"].setdefault(name, {})[split] = summarize(rows)
            (out / f"{split}_per_site_{name}.json").write_text(
                json.dumps(per_group(rows), ensure_ascii=False, indent=2), encoding="utf-8")
            rows = []
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
