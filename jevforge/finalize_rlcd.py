"""Finalize an interrupted RLCD run from its crash-safe best checkpoint."""
import argparse
import json
import os
import shutil
from pathlib import Path

from .rlcd import evaluate_policy, load_checkpoint
from .schema import SPLITS
from .train import (build_examples, fit_temperature, load_env_file,
                    load_records_tolerant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-questions", type=int, default=300)
    parser.add_argument("--microbatch-tokens", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--wandb-run-id", default=None)
    parser.add_argument("--wandb-project", default="jevforge")
    args = parser.parse_args()

    load_env_file()
    import random
    import torch
    from safetensors.torch import load_file

    device = torch.device("cuda:0")
    model, tokenizer, source_config, backbone_dir = load_checkpoint(
        args.source_checkpoint, device)
    state = load_file(str(Path(args.output_dir) / "best.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    records_path = Path(args.records)
    files = ([records_path] if records_path.is_file() else
             [records_path / f"{split}.jsonl" for split in SPLITS])
    splits = {}
    for file in files:
        if file.exists():
            for record in load_records_tolerant(file):
                splits.setdefault(record["split"], []).append(record)
    max_length = int(source_config.get("max_length", 768))
    data = {split: build_examples(records, tokenizer, max_length)
            for split, records in splits.items()}

    temperature = fit_temperature(
        model, data.get("calibration") or [], tokenizer.pad_token_id,
        device, True, args.microbatch_tokens, per_type=True)

    # The reference is only needed for post-calibration KL diagnostics. Loading
    # it after temperature fitting keeps calibration memory to one model.
    reference, _, _, _ = load_checkpoint(args.source_checkpoint, device)
    reference.eval()

    def subset(examples, limit, salt):
        if not limit or len(examples) <= limit:
            return examples
        order = list(range(len(examples)))
        random.Random(args.seed + salt).shuffle(order)
        return [examples[index] for index in order[:limit]]

    best = json.loads((Path(args.output_dir) / "best.json").read_text())
    final = {"best_step": best["step"], "best_dev_score": best["dev_score"],
             "temperature": temperature, "splits": {}}
    for split, salt in (("test", 211), ("ood", 307)):
        if data.get(split):
            final["splits"][split] = evaluate_policy(
                model, reference, subset(data[split], args.final_questions, salt),
                tokenizer.pad_token_id, device, True, args.microbatch_tokens)

    out = Path(args.output_dir)
    shutil.copytree(backbone_dir, out / "backbone", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pth"))
    (out / "config.json").write_text(json.dumps({
        "schema": "jevforge-checkpoint-v1", "base_model": source_config["base_model"],
        "hidden_size": source_config["hidden_size"], "max_length": max_length,
        "temperature": temperature, "best_step": best["step"],
        "source_checkpoint": str(Path(args.source_checkpoint).resolve()),
        "rlcd": {"algorithm": "independent-rlcd-style-v1", "seed": args.seed},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "rlcd_final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2),
                                         encoding="utf-8")

    if args.wandb_run_id:
        import wandb
        run = wandb.init(project=args.wandb_project, id=args.wandb_run_id,
                         resume="allow")
        run.log({f"final/{split}/{key}": value
                 for split, block in final["splits"].items()
                 for key, value in block.items() if isinstance(value, (int, float))})
        run.summary.update({"best_step": best["step"],
                            "best_dev_score": best["dev_score"],
                            "temperature": temperature})
        run.finish()
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
