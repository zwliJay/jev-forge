"""Migrate completed local training runs to SwanLab (project jev-forge).

Only touches runs marked finished (best.json + config.json present and the
train process is gone). Requires a prior `swanlab login` on this host.
"""
import argparse
import json
from pathlib import Path

COMPLETED = {  # run-dir -> experiment name on SwanLab
    "runs/full_v2": "full-v2-compile-dyn",
    "runs/forge_tune": "forge-tune-final",
}


def migrate(run_dir: Path, name: str, project: str, workspace: str):
    import swanlab

    metrics_file = run_dir / "train_metrics.jsonl"
    log_file = run_dir / "train_log.json"
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    swanlab.init(project=project, workspace=workspace, experiment_name=name,
                 config={"args": config.get("args", {}),
                         "best_step": config.get("best_step"),
                         "temperature": config.get("temperature"),
                         "base_model": config.get("base_model")})
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            step = row.pop("step", None)
            swanlab.log({f"train/{k}": v for k, v in row.items()}, step=step)
    if log_file.exists():
        for entry in json.loads(log_file.read_text(encoding="utf-8")):
            payload = {k: v for k, v in entry.items() if k != "step"}
            swanlab.log({f"dev/{k}": v for k, v in payload.items()},
                        step=entry.get("step"))
    swanlab.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--project", default="jev-forge")
    parser.add_argument("--workspace", default="zwli")
    parser.add_argument("--runs", nargs="*",
                        default=[f"{k}:{v}" for k, v in COMPLETED.items()])
    args = parser.parse_args()
    repo = Path(args.repo)
    import subprocess

    for spec in args.runs:
        run_dir, _, name = spec.partition(":")
        if not (repo / run_dir / "best.json").exists():
            print(f"skip {run_dir}: not finished")
            continue
        alive = subprocess.run(["pgrep", "-f", f"jevforge.train.*{run_dir.split('/')[-1]}"],
                               capture_output=True).returncode == 0
        if alive:
            print(f"skip {run_dir}: still training")
            continue
        print(f"migrating {run_dir} as {name}")
        migrate(repo / run_dir, name, args.project, args.workspace)


if __name__ == "__main__":
    main()
