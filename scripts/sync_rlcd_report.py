"""Attach a full supervised-vs-RLCD benchmark report to an existing W&B run."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def flatten(report):
    values = {}
    for arm, splits in report["runs"].items():
        for split, block in splits.items():
            values[f"benchmark/{arm}/{split}/questions"] = block["questions"]
            for kind, metrics in block.items():
                if kind == "questions":
                    continue
                for metric, value in metrics.items():
                    values[f"benchmark/{arm}/{split}/{kind}/{metric}"] = value
    return values


def deltas(report):
    values = {}
    supervised = report["runs"]["supervised"]
    rlcd = report["runs"]["rlcd"]
    for split in ("test", "ood"):
        for kind, metric in (("choice", "top1_in_positives"),
                             ("choice", "mass_on_positives"),
                             ("noul", "accuracy"), ("noul", "brier"),
                             ("noul", "ece"), ("score", "mae")):
            values[f"delta/{split}/{kind}/{metric}"] = (
                rlcd[split][kind][metric] - supervised[split][kind][metric])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--project", default="jevforge")
    parser.add_argument("--artifact-name", default=None)
    args = parser.parse_args()

    from jevforge.train import load_env_file
    load_env_file()
    import wandb
    report_path = Path(args.report).resolve(strict=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = {**flatten(report), **deltas(report)}
    run = wandb.init(project=args.project, id=args.run_id, resume="allow")
    run.log(metrics)
    run.summary.update(metrics)
    artifact = wandb.Artifact(
        args.artifact_name or f"rlcd-evaluation-{args.run_id}", type="evaluation",
        metadata={"test_questions": report["runs"]["rlcd"]["test"]["questions"],
                  "ood_questions": report["runs"]["rlcd"]["ood"]["questions"]})
    artifact.add_file(str(report_path), name="metrics.json")
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    main()
