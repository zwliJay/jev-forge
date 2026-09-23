"""Record the checkpoint operating the fixed mock site (web/demo_dom.json).

Runs the real model on every step of every task and writes
web/demo_recording.json with the complete distribution, its pick and the
verdict — so the static site demo replays genuine model behaviour.
"""
import argparse
import json
from pathlib import Path

ACTION_QUESTION = {
    "type": "choice",
    "instructions": "Which page element should be interacted with next to make progress on the task?",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dom", default="web/demo_dom.json")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", default="web/demo_recording.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    dom = json.loads(Path(args.dom).read_text(encoding="utf-8"))
    from .predict import Predictor

    predictor = Predictor(args.checkpoint_dir, device_name=args.device)
    tasks = []
    for task in dom["tasks"]:
        steps = []
        for step in task["steps"]:
            state = json.dumps(
                {"elements": [{"id": e["id"], "text": e["text"]} for e in step["elements"]],
                 "operation": step.get("hint", ""), "task": task["instruction"]},
                ensure_ascii=False, sort_keys=True)
            question = dict(ACTION_QUESTION,
                            criteria={e["id"]: e["text"] for e in step["elements"]})
            answer = predictor.decide(state, {"action": question})["action"]
            steps.append({
                "url": step["url"], "elements": step["elements"], "gold": step["gold"],
                "probabilities": answer["probabilities"], "chosen": answer["choice"],
                "correct": answer["choice"] in step["gold"],
                "confidence": answer["confidence"],
            })
        tasks.append({"id": task["id"], "instruction": task["instruction"], "steps": steps})
    output = Path(args.output)
    output.write_text(json.dumps({
        "meta": {"site": dom["site"]["name"], "domain": dom["site"]["domain"],
                 "model": "jevforge-qwen3-0.6b", "replay": True},
        "tasks": tasks,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tasks": len(tasks),
                      "steps": sum(len(t["steps"]) for t in tasks),
                      "on_target": sum(s["correct"] for t in tasks for s in t["steps"])}))


if __name__ == "__main__":
    main()
