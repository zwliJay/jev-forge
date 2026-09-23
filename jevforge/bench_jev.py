"""Head-to-head: real Jev vs JevForge on the fixed held-out splits.

Sends each record's full question set (action choice + is_target noul +
difficulty score) to the real Jev model via a configured decisions endpoint,
one request per record, then scores answers with the same conventions as
jevforge.evaluate.
"""
import argparse
import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DECISIONS = os.environ.get("JEVFORGE_DECISIONS_ENDPOINT", "")
JEV_MODEL = "~typesafe/jev-latest"


def post(url, body, api_key, timeout=120):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def ask_jev(record, api_key):
    request = record["request"]
    answer = post(DECISIONS, {"model": JEV_MODEL, "state": request["state"],
                              "questions": request["questions"]}, api_key)["answers"]
    return answer


def score_choice(record, probabilities):
    target = record["targets"]["action"]
    ids = list(record["request"]["questions"]["action"]["criteria"])
    probs = [float(probabilities.get(i, 0.0)) for i in ids]
    total = math.fsum(probs) or 1.0
    probs = [p / total for i, p in zip(ids, probs)]
    positives = {i for i in ids if target[i] > 0}
    top = max(range(len(ids)), key=probs.__getitem__)
    ce = -math.fsum(target[i] * math.log(max(p, 1e-12))
                    for i, p in zip(ids, probs))
    return {"top1_in_positives": ids[top] in positives,
            "exact_gold": ids[top] == record["gold"]["action"],
            "mass_on_positives": math.fsum(p for i, p in zip(ids, probs) if i in positives),
            "ce": ce}


def score_noul(record, p_true):
    truth = record["targets"]["is_target"]["true"]
    return {"accuracy": (p_true >= 0.5) == (truth >= 0.5),
            "brier": (p_true - truth) ** 2}


def score_difficulty(record, probabilities):
    target = record["targets"]["difficulty"]
    expected_p = math.fsum(int(k) * v for k, v in probabilities.items())
    expected_q = math.fsum(int(k) * v for k, v in target.items())
    ce = -math.fsum(v * math.log(max(probabilities.get(k, 1e-12), 1e-12))
                    for k, v in target.items())
    return {"mae": abs(expected_p - expected_q), "ce": ce}


def mean(values):
    return round(math.fsum(values) / max(len(values), 1), 4)


def summarize(rows):
    out = {}
    if rows.get("choice"):
        out["choice"] = {k: mean([r[k] for r in rows["choice"]]) for k in
                         ("top1_in_positives", "exact_gold", "mass_on_positives", "ce")}
    if rows.get("noul"):
        out["noul"] = {k: mean([r[k] for r in rows["noul"]]) for k in ("accuracy", "brier")}
    if rows.get("score"):
        out["score"] = {k: mean([r[k] for r in rows["score"]]) for k in ("mae", "ce")}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="dataset dir with <split>.jsonl")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--api-key-env", default="JEVFORGE_API_KEY")
    parser.add_argument("--output", default="artifacts/jev_comparison.json")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    report = {"jev_model": JEV_MODEL, "splits": {}}
    for split in args.splits.split(","):
        records = [json.loads(line) for line in
                   (Path(args.records) / f"{split}.jsonl").read_text(
                       encoding="utf-8").splitlines()][:args.limit]
        rows = {"choice": [], "noul": [], "score": []}
        def work(record):
            try:
                answer = ask_jev(record, api_key)
                choice = score_choice(record, answer["action"]["probabilities"])
                noul = score_noul(record, answer["is_target"]["noul"])
                score = score_difficulty(record, answer["difficulty"]["probabilities"])
                return (choice, noul, score)
            except Exception:  # noqa: BLE001
                return None

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for result in pool.map(work, records):
                if result is None:
                    continue
                choice, noul, score = result
                rows["choice"].append(choice)
                rows["noul"].append(noul)
                rows["score"].append(score)
        entry = {"questions": len(rows["choice"]), "jev": summarize(rows)}
        report["splits"][split] = entry

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
