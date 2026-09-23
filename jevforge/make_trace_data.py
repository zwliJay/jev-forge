"""Build trajectory replay data for the web trace demo.

Groups held-out records by Mind2Web annotation (one multi-step task), replays
the checkpoint on every step, and emits web/trace_data.json: a Jev-site-style
player where you watch the model pick its way through a real task, step by
step, on pages it never trained on.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="dataset dir with <split>.jsonl")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--trajectories", type=int, default=8)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    groups = defaultdict(list)
    task_of = {}
    for split in args.splits.split(","):
        path = Path(args.records) / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            annotation = (record.get("meta") or {}).get("annotation_id")
            if not annotation:
                continue
            groups[annotation].append(record)
            task_of[annotation] = json.loads(record["request"]["state"]).get("task", "")

    ranked = sorted((g for g in groups.values() if len(g) >= args.min_steps),
                    key=lambda g: -len(g))[:args.trajectories]
    if not ranked:
        raise SystemExit("no multi-step annotations found in the given splits")

    from .predict import Predictor

    predictor = Predictor(args.checkpoint_dir, device_name=args.device)
    trajectories = []
    for group in ranked:
        group.sort(key=lambda r: r["id"])
        steps = []
        for record in group:
            state = json.loads(record["request"]["state"])
            examples_probs = predictor.decide(
                record["request"]["state"],
                {"action": record["request"]["questions"]["action"]})
            answer = examples_probs["action"]
            gold = record["meta"]["positive_ids"]
            steps.append({
                "elements": state["elements"],
                "gold": gold,
                "probabilities": answer["probabilities"],
                "chosen": answer["choice"],
                "correct": answer["choice"] in gold,
                "confidence": answer["confidence"],
                "human_action": record["meta"].get("target_action_reprs", ""),
                "operation": state.get("operation", ""),
            })
        trajectories.append({
            "annotation_id": group[0]["meta"]["annotation_id"],
            "website": group[0]["source_group"],
            "split": group[0]["split"],
            "task": task_of[group[0]["meta"]["annotation_id"]],
            "steps": steps,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "meta": {"model": "jevforge-qwen3.5-0.8b", "replay": True,
                 "note": "held-out Mind2Web annotations; gold = human-annotated "
                         "positive elements; chosen = model top-1"},
        "trajectories": trajectories,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"trajectories": len(trajectories),
                      "steps": sum(len(t["steps"]) for t in trajectories)}))


if __name__ == "__main__":
    main()
