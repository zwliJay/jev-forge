"""RLCD-style policy optimisation for finite, typed browser decisions.

This is an independent research baseline, not TypeSafe's unpublished recipe.
It starts from a supervised JevForge checkpoint and optimises three observable
properties of a complete candidate distribution:

* task utility: put probability mass on valid browser actions;
* calibration: minimise a proper Brier score against the target distribution;
* stability: stay close to the supervised reference policy with forward KL.

The utility term can be computed exactly over the finite action set or with a
group-sampled REINFORCE estimator. Exact mode is the low-variance default;
sampled mode is useful for studying the sampler/optimiser interaction.
"""
import argparse
import copy
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch

from .model import JevForgeModel, backbone_hidden_size
from .train import (batches_by_tokens, build_examples, fit_temperature,
                    forward_microbatch, load_env_file, load_records_tolerant)
from .schema import SPLITS


def utility_rewards(target):
    """Map target mass to [0, 1], preserving ties between valid actions."""
    peak = target.max().clamp_min(1e-8)
    return target / peak


def distribution_metrics(logits, target, reference_logits=None):
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    ce = -(target * log_probs).sum()
    brier = ((probs - target) ** 2).sum()
    entropy = -(probs * log_probs).sum()
    utility = (probs * utility_rewards(target)).sum()
    if reference_logits is None:
        kl = logits.new_zeros(())
    else:
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        kl = (probs * (log_probs - reference_log_probs)).sum()
    return {
        "probs": probs,
        "log_probs": log_probs,
        "ce": ce,
        "brier": brier,
        "entropy": entropy,
        "utility": utility,
        "kl": kl,
    }


def rlcd_loss(logits, target, reference_logits, *, mode="exact", group_size=8,
              utility_weight=1.0, calibration_weight=0.5, kl_weight=0.02,
              entropy_weight=0.0, generator=None):
    """Return the differentiable objective and detached diagnostic metrics."""
    metrics = distribution_metrics(logits, target, reference_logits)
    if mode == "exact":
        policy_loss = -metrics["utility"]
        sampled_reward = metrics["utility"].detach()
    elif mode == "sampled":
        actions = torch.multinomial(metrics["probs"].detach(), group_size,
                                    replacement=True, generator=generator)
        rewards = utility_rewards(target)[actions]
        advantages = rewards - rewards.mean()
        policy_loss = -(advantages.detach() * metrics["log_probs"][actions]).mean()
        sampled_reward = rewards.mean().detach()
    else:
        raise ValueError(f"unknown RLCD mode: {mode}")

    loss = (utility_weight * policy_loss
            + calibration_weight * metrics["brier"]
            + kl_weight * metrics["kl"]
            - entropy_weight * metrics["entropy"])
    diagnostics = {
        "loss": loss.detach(),
        "policy_loss": policy_loss.detach(),
        "utility": metrics["utility"].detach(),
        "sampled_reward": sampled_reward,
        "brier": metrics["brier"].detach(),
        "ce": metrics["ce"].detach(),
        "kl": metrics["kl"].detach(),
        "entropy": metrics["entropy"].detach(),
    }
    return loss, diagnostics


@torch.no_grad()
def evaluate_policy(model, reference, examples, pad_id, device, autocast,
                    micro_tokens):
    model.eval()
    reference.eval()
    totals = {key: 0.0 for key in
              ("ce", "brier", "utility", "kl", "entropy", "top1")}
    per_type = {}
    count = 0
    for _, micro in batches_by_tokens(examples, 24, micro_tokens):
        current = forward_microbatch(model, micro, pad_id, device, autocast)
        baseline = forward_microbatch(reference, micro, pad_id, device, autocast)
        for example, logits, ref_logits in zip(micro, current, baseline):
            target = torch.tensor(example["target"], device=device)
            values = distribution_metrics(logits, target, ref_logits)
            row = {key: float(values[key]) for key in
                   ("ce", "brier", "utility", "kl", "entropy")}
            row["top1"] = float(target[logits.argmax()] > 0)
            for key, value in row.items():
                totals[key] += value
            kind = per_type.setdefault(example["type"],
                                       {"count": 0, **{k: 0.0 for k in row}})
            kind["count"] += 1
            for key, value in row.items():
                kind[key] += value
            count += 1
    model.train()
    out = {key: value / max(count, 1) for key, value in totals.items()}
    out["questions"] = count
    out["per_type"] = {
        kind: {key: value / max(block["count"], 1)
               for key, value in block.items() if key != "count"}
        for kind, block in per_type.items()
    }
    return out


def load_checkpoint(checkpoint_dir, device):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    root = Path(checkpoint_dir).resolve(strict=True)
    run_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    backbone_dir = root / "backbone"
    config = AutoConfig.from_pretrained(str(backbone_dir), local_files_only=True,
                                        trust_remote_code=False)
    config.use_cache = False
    backbone = AutoModel.from_config(config, attn_implementation="sdpa",
                                     trust_remote_code=False)
    model = JevForgeModel(backbone, backbone_hidden_size(config))
    state = load_file(str(root / "best.safetensors"), device="cpu")
    state = {key.replace("._orig_mod.", "."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(str(backbone_dir), local_files_only=True,
                                               trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, run_config, backbone_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("exact", "sampled"), default="exact")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--questions-per-batch", type=int, default=6)
    parser.add_argument("--microbatch-tokens", type=int, default=10000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--head-learning-rate", type=float, default=5e-5)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--calibration-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=0.02)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--dev-questions", type=int, default=300,
                        help="fixed seeded dev subset used for checkpoint selection; 0 = full")
    parser.add_argument("--final-questions", type=int, default=300,
                        help="fixed seeded test/OOD subset for policy diagnostics; 0 = full")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="jevforge")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    load_env_file()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    model, tokenizer, source_config, backbone_dir = load_checkpoint(
        args.checkpoint_dir, device)
    reference = copy.deepcopy(model).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    tracker = None
    if args.wandb:
        import wandb
        tracker = wandb.init(project=args.wandb_project,
                             name=args.run_name or Path(args.output_dir).name,
                             config={**vars(args), "source_checkpoint": str(args.checkpoint_dir),
                                     "algorithm": "independent-rlcd-style-v1"})

    path = Path(args.records)
    files = [path] if path.is_file() else [path / f"{split}.jsonl" for split in SPLITS]
    splits = {}
    for file in files:
        if file.exists():
            for record in load_records_tolerant(file):
                splits.setdefault(record["split"], []).append(record)
    max_length = int(source_config.get("max_length", 768))
    data = {split: build_examples(records, tokenizer, max_length)
            for split, records in splits.items()}
    train_pool = data.get("train") or []
    if not train_pool:
        raise SystemExit("no train questions")

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.learning_rate},
        {"params": model.head.parameters(), "lr": args.head_learning_rate},
    ], weight_decay=0.01)
    schedule = batches_by_tokens(train_pool, args.questions_per_batch,
                                 args.microbatch_tokens)
    rng = random.Random(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_file = (out / "rlcd_metrics.jsonl").open("w", encoding="utf-8")
    best_score, best_state, best_step = float("inf"), None, 0

    def fixed_subset(examples, limit, salt):
        if not limit or len(examples) <= limit:
            return examples
        order = list(range(len(examples)))
        random.Random(args.seed + salt).shuffle(order)
        return [examples[index] for index in order[:limit]]

    dev_examples = fixed_subset(data.get("dev") or train_pool[:200],
                                args.dev_questions, 101)
    step = 0
    while step < args.steps:
        rng.shuffle(schedule)
        for _, micro in schedule:
            if step >= args.steps:
                break
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                reference_logits = forward_microbatch(
                    reference, micro, tokenizer.pad_token_id, device, True)
            logits = forward_microbatch(model, micro, tokenizer.pad_token_id,
                                        device, True)
            losses, sums = [], {}
            for example, current, baseline in zip(micro, logits, reference_logits):
                target = torch.tensor(example["target"], device=device)
                loss, diagnostic = rlcd_loss(
                    current, target, baseline, mode=args.mode,
                    group_size=args.group_size,
                    utility_weight=args.utility_weight,
                    calibration_weight=args.calibration_weight,
                    kl_weight=args.kl_weight,
                    entropy_weight=args.entropy_weight,
                    generator=generator)
                losses.append(loss)
                for key, value in diagnostic.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            payload = {key: value / len(losses) for key, value in sums.items()}
            payload.update({"step": step, "grad_norm": float(grad_norm),
                            "questions": len(micro),
                            "seconds": time.perf_counter() - started})
            metrics_file.write(json.dumps(payload) + "\n")
            metrics_file.flush()
            if tracker:
                tracker.log({f"train/{k}": v for k, v in payload.items()
                             if k != "step"}, step=step)
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate_policy(model, reference,
                                      dev_examples,
                                      tokenizer.pad_token_id, device, True,
                                      args.microbatch_tokens)
                print(json.dumps({"step": step, "dev": dev}), flush=True)
                if tracker:
                    tracker.log({f"dev/{k}": v for k, v in dev.items()
                                 if isinstance(v, (int, float))}, step=step)
                score = dev["ce"] + args.calibration_weight * dev["brier"]
                if score < best_score:
                    best_score, best_step = score, step
                    best_state = {key: value.detach().cpu().clone()
                                  for key, value in model.state_dict().items()}
                    from safetensors.torch import save_file
                    save_file(best_state, str(out / "best.safetensors"))
                    (out / "best.json").write_text(json.dumps({
                        "step": best_step, "dev_score": best_score,
                        "dev_questions": len(dev_examples),
                    }), encoding="utf-8")
    metrics_file.close()
    if best_state is None:
        best_state = {key: value.detach().cpu().clone()
                      for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    temperature = fit_temperature(model, data.get("calibration") or [],
                                  tokenizer.pad_token_id, device, True,
                                  args.microbatch_tokens, per_type=True)
    final = {"best_step": best_step, "best_dev_score": best_score,
             "temperature": temperature, "splits": {}}
    for split in ("test", "ood"):
        if data.get(split):
            final["splits"][split] = evaluate_policy(
                model, reference,
                fixed_subset(data[split], args.final_questions,
                             211 if split == "test" else 307),
                tokenizer.pad_token_id,
                device, True, args.microbatch_tokens)
    from safetensors.torch import save_file
    save_file(best_state, str(out / "best.safetensors"))
    shutil.copytree(backbone_dir, out / "backbone", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pth"))
    (out / "config.json").write_text(json.dumps({
        "schema": "jevforge-checkpoint-v1", "base_model": source_config["base_model"],
        "hidden_size": source_config["hidden_size"], "max_length": max_length,
        "temperature": temperature, "best_step": best_step,
        "rlcd": vars(args), "source_checkpoint": str(Path(args.checkpoint_dir).resolve()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "rlcd_final.json").write_text(json.dumps(final, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    if tracker:
        tracker.log({f"final/{split}/{key}": value
                     for split, block in final["splits"].items()
                     for key, value in block.items() if isinstance(value, (int, float))},
                    step=step)
        tracker.summary.update({"best_step": best_step,
                                "best_dev_score": best_score})
        tracker.finish()
    print(json.dumps(final, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
