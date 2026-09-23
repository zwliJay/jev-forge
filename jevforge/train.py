"""JevForge trainer: complete-question distribution learning.

Loss per question = cross-entropy of the candidate softmax against the
target distribution plus an optional vector-Brier term. Every question is
one supervision unit: its whole candidate set enters one softmax; candidates
are never split across normalisation groups. A short head-only warmup
precedes full-model updates; the checkpoint with the lowest dev CE wins.
After training, a scalar temperature is fitted on the calibration split.
"""
import argparse
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
from torch.nn.functional import cross_entropy

from .encode import encode_examples
from .model import JevForgeModel, backbone_hidden_size
from .schema import SPLITS, load_records, validate_distribution, candidate_ids


def load_env_file(path=".env"):
    """Tiny .env reader: KEY=VALUE lines only, never overrides real env."""
    file = Path(path)
    if not file.exists():
        return
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def load_records_tolerant(path):
    """Stream-friendly reader: skip torn tail lines and invalid records."""
    from .schema import validate_record

    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(validate_record(json.loads(line)))
        except (ValueError, KeyError, TypeError):
            continue
    return records


def build_examples(records, tokenizer, max_length, label_smoothing=0.0):
    examples = []
    for record in records:
        kinds = record.get("target_kinds", {})
        for example in encode_examples(record["request"], tokenizer, max_length):
            ids = example["candidate_ids"]
            target = validate_distribution(record["targets"][example["question_id"]],
                                           ids, f"{record['id']}:{example['question_id']}")
            if label_smoothing and kinds.get(example["question_id"]) in (
                    "deterministic_truth", "uniform_over_positive_elements"):
                k = len(ids)
                target = [(1 - label_smoothing) * q + label_smoothing / k for q in target]
            example["record_id"] = record["id"]
            example["source_group"] = record["source_group"]
            example["target"] = target
            examples.append(example)
    return examples


def batches_by_tokens(examples, questions_per_batch, max_tokens):
    """Group complete questions into microbatches, length-bucketed to cut padding."""
    order = sorted(range(len(examples)),
                   key=lambda i: max(len(leaf) for leaf in examples[i]["leaves"]))
    out = []
    for start in range(0, len(order), questions_per_batch):
        chunk = [examples[i] for i in order[start:start + questions_per_batch]]
        micro, used, questions = [], 0, []
        for example in chunk:
            cost = max(len(leaf) for leaf in example["leaves"]) * len(example["leaves"])
            if micro and used + cost > max_tokens:
                out.append((questions, micro))
                micro, used, questions = [], 0, []
            micro.append(example)
            used += cost
            questions.append(example)
        if micro:
            out.append((questions, micro))
    return out


def forward_microbatch(model, micro, pad_id, device, autocast):
    leaves, owner = [], []
    for qi, example in enumerate(micro):
        for leaf in example["leaves"]:
            leaves.append(leaf)
            owner.append(qi)
    width = max(len(leaf) for leaf in leaves)
    width = (width + 63) // 64 * 64  # pad to 64-token buckets for stable shapes
    input_ids = torch.full((len(leaves), width), pad_id, dtype=torch.long)
    attention = torch.zeros((len(leaves), width), dtype=torch.long)
    for i, leaf in enumerate(leaves):
        input_ids[i, :len(leaf)] = torch.tensor(leaf, dtype=torch.long)
        attention[i, :len(leaf)] = 1
    input_ids, attention = input_ids.to(device), attention.to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = model(input_ids, attention).float()
    owner = torch.tensor(owner, device=device)
    grouped = [logits[owner == qi] for qi in range(len(micro))]
    return grouped


def question_loss(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    ce = -(target * log_probs).sum()
    probs = logits.softmax(dim=-1)
    brier = ((probs - target) ** 2).sum()
    return ce, brier


@torch.no_grad()
def evaluate_split(model, examples, pad_id, device, autocast, micro_tokens):
    """Mean CE overall and per question type."""
    model.eval()
    totals, counts = 0.0, {"all": 0}
    for questions, micro in batches_by_tokens(examples, 24, micro_tokens):
        grouped = forward_microbatch(model, micro, pad_id, device, autocast)
        for example, logits in zip(micro, grouped):
            target = torch.tensor(example["target"], device=device)
            log_probs = torch.log_softmax(logits, dim=-1)
            ce = -(target * log_probs).sum().item()
            totals += ce
            counts["all"] += 1
            counts[example["type"]] = counts.get(example["type"], 0) + 1
            counts.setdefault(f"ce_{example['type']}", 0.0)
            counts[f"ce_{example['type']}"] += ce
    model.train()
    out = {"ce": totals / max(counts["all"], 1)}
    for kind in ("choice", "noul", "score"):
        if counts.get(kind):
            out[f"ce_{kind}"] = counts[f"ce_{kind}"] / counts[kind]
    return out


@torch.no_grad()
def fit_temperature(model, examples, pad_id, device, autocast, micro_tokens, per_type=False):
    """Grid+refine search for the temperature minimising calibration NLL."""
    cached = []
    for questions, micro in batches_by_tokens(examples, 24, micro_tokens):
        for example, logits in zip(micro, forward_microbatch(model, micro, pad_id, device, autocast)):
            cached.append((logits.detach(), torch.tensor(example["target"], device=device),
                           example["type"]))

    def search(items):
        if not items:
            return 1.0

        # Evaluate a whole temperature grid on-device. The former scalar loop
        # called .item() once per example and grid point, creating hundreds of
        # thousands of GPU synchronisations on full calibration splits.
        def grid_search(temperatures):
            losses = torch.zeros_like(temperatures)
            for logits, target, _ in items:
                scaled = logits.unsqueeze(0) / temperatures.unsqueeze(1)
                losses += -(target.unsqueeze(0)
                            * torch.log_softmax(scaled, dim=-1)).sum(dim=-1)
            return int(losses.argmin().item())

        device = items[0][0].device
        coarse = torch.linspace(0.05, 6.0, 120, device=device)
        coarse_index = grid_search(coarse)
        center = float(coarse[coarse_index])
        lo, hi = max(0.05, center - 0.05), min(6.0, center + 0.05)
        fine = torch.linspace(lo, hi, 201, device=device)
        fine_index = grid_search(fine)
        return round(float(fine[fine_index]), 4)

    if per_type:
        return {kind: search([c for c in cached if c[2] == kind])
                for kind in ("choice", "noul", "score")}
    return search(cached)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="all.jsonl or a split directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", required=True, help="local Hugging Face snapshot directory")
    parser.add_argument(
        "--backbone-init",
        choices=["scratch", "pretrained"],
        default="scratch",
        help="scratch preserves the 0.6B v1 recipe; pretrained loads upstream weights",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--head-steps", type=int, default=12)
    parser.add_argument("--questions-per-batch", type=int, default=12)
    parser.add_argument("--microbatch-tokens", type=int, default=24000)
    parser.add_argument("--brier-weight", type=float, default=0.5)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr-schedule", choices=["constant", "cosine"], default="cosine",
                        help="cosine = linear warmup (head steps) then cosine decay")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="mix deterministic one-hot targets with uniform: (1-e)*q + e/K")
    parser.add_argument("--per-type-temperature", action="store_true",
                        help="fit a separate calibration temperature per question type")
    parser.add_argument("--compile", action="store_true", help="torch.compile the backbone")
    parser.add_argument("--compile-dynamic", action="store_true",
                        help="compile with dynamic shapes (no per-shape recompile storms)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="disable gradient checkpointing (faster when memory allows)")
    parser.add_argument("--wandb", action="store_true",
                        help="log metrics to Weights & Biases (needs WANDB_API_KEY or .env)")
    parser.add_argument("--swanlab", action="store_true",
                        help="also log to SwanLab (needs prior `swanlab login`)")
    parser.add_argument("--wandb-project", default="jevforge")
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    load_env_file()
    tracker = None
    if args.wandb:
        try:
            import wandb

            tracker = wandb.init(project=args.wandb_project,
                                 name=args.run_name or Path(args.output_dir).name,
                                 config=vars(args))
        except Exception as exc:  # noqa: BLE001 — training must never depend on wandb
            print(json.dumps({"wandb": "disabled", "reason": str(exc)[:120]}), flush=True)
    swan = None
    if args.swanlab:
        try:
            import swanlab

            swan = swanlab.init(project="jev-forge", workspace="zwli",
                                experiment_name=args.run_name
                                or Path(args.output_dir).name,
                                config=vars(args))
        except Exception as exc:  # noqa: BLE001 — same rule for swanlab
            print(json.dumps({"swanlab": "disabled", "reason": str(exc)[:120]}), flush=True)

    def track(payload, step):
        if tracker:
            tracker.log(payload, step=step)
        if swan:
            import swanlab as _swanlab

            _swanlab.log(payload, step=step)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    autocast = True

    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone_config = AutoConfig.from_pretrained(args.base_model, local_files_only=True,
                                                 trust_remote_code=False)
    backbone_config.use_cache = False
    if args.backbone_init == "pretrained":
        backbone = AutoModel.from_pretrained(
            args.base_model,
            local_files_only=True,
            trust_remote_code=False,
            attn_implementation="sdpa",
            dtype=torch.bfloat16,
        )
    else:
        backbone = AutoModel.from_config(
            backbone_config,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )
    print(json.dumps({
        "backbone_init": args.backbone_init,
        "model_type": backbone_config.model_type,
        "name_or_path": str(args.base_model),
        "parameters": sum(parameter.numel() for parameter in backbone.parameters()),
        "dtype": str(next(backbone.parameters()).dtype),
    }), flush=True)
    hidden_size = backbone_hidden_size(backbone_config)
    model = JevForgeModel(backbone, hidden_size).to(device)
    if not args.no_grad_ckpt:
        model.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.compile:
        model.backbone = torch.compile(model.backbone,
                                       dynamic=True if args.compile_dynamic else None)
    model.train()

    path = Path(args.records)
    files = [path] if path.is_file() else [path / f"{s}.jsonl" for s in SPLITS
                                           if (path / f"{s}.jsonl").exists()]
    splits = {}
    for file in files:
        for record in load_records_tolerant(file):
            splits.setdefault(record["split"], []).append(record)
    data = {split: build_examples(records, tokenizer, args.max_length,
                                  args.label_smoothing)
            for split, records in splits.items()}
    train_pool = data.get("train") or []
    if not train_pool:
        raise SystemExit("no train questions")
    print({s: len(v) for s, v in data.items()}, flush=True)

    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)

    try:
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ], weight_decay=0.01, fused=True)
    except (RuntimeError, TypeError):
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ], weight_decay=0.01)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file  # noqa: F811 — best ckpt saved eagerly

    def save_best(state, ce, at_step):
        cleaned = {k.replace("._orig_mod.", "."): v for k, v in state.items()}
        save_file(cleaned, str(out / "best.safetensors"))
        (out / "best.json").write_text(json.dumps({"step": at_step, "dev_ce": ce}),
                                      encoding="utf-8")

    metrics_file = (out / "train_metrics.jsonl").open("w", encoding="utf-8")
    best_ce, best_state, best_step = float("inf"), None, 0
    schedule = batches_by_tokens(train_pool, args.questions_per_batch,
                                 args.microbatch_tokens)
    step = 0
    rng = random.Random(args.seed)
    log = []
    torch.cuda.reset_peak_memory_stats()
    while step < args.steps:
        rng.shuffle(schedule)
        for questions, micro in schedule:
            if step >= args.steps:
                break
            started = time.perf_counter()
            if step == args.head_steps:
                for parameter in model.backbone.parameters():
                    parameter.requires_grad_(True)
            optimizer.zero_grad(set_to_none=True)
            grouped = forward_microbatch(model, micro, tokenizer.pad_token_id,
                                         device, autocast)
            losses, ce_sum, brier_sum = [], 0.0, 0.0
            for example, logits in zip(micro, grouped):
                target = torch.tensor(example["target"], device=device)
                ce, brier = question_loss(logits, target)
                ce_sum += ce.item()
                brier_sum += brier.item()
                losses.append(ce + args.brier_weight * brier)
            loss = torch.stack(losses).mean()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            current_lr = args.backbone_lr
            if args.lr_schedule == "cosine":
                warmup = max(args.head_steps, 1)
                progress = min(step / max(args.steps - 1, 1), 1.0)
                decay = 0.5 * (1 + math.cos(math.pi * progress))
                scale = min(step / warmup, 1.0) * decay
                optimizer.param_groups[0]["lr"] = args.backbone_lr * scale
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = args.head_lr * scale
                current_lr = args.backbone_lr * scale
            optimizer.step()
            step += 1
            step_metrics = {
                "step": step,
                "loss": loss.item(),
                "loss_ce": ce_sum / max(1, len(losses)),
                "loss_brier": brier_sum / max(1, len(losses)),
                "grad_norm": float(grad_norm),
                "lr_backbone": current_lr,
                "questions": len(micro),
                "tokens": sum(len(leaf) for ex in micro for leaf in ex["leaves"]),
                "sec": round(time.perf_counter() - started, 4),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            }
            metrics_file.write(json.dumps(step_metrics) + "\n")
            metrics_file.flush()
            track({f"train/{k}": v for k, v in step_metrics.items()
                   if k != "step"}, step)
            if step % args.eval_every == 0 or step == args.steps:
                dev = evaluate_split(model, data.get("dev") or data["train"][:200],
                                     tokenizer.pad_token_id, device, autocast,
                                     args.microbatch_tokens)
                log.append({"step": step, "loss": loss.item(), **dev})
                print(json.dumps(log[-1]), flush=True)
                track({f"dev/{k}": v for k, v in dev.items()}, step)
                if dev["ce"] < best_ce:
                    best_ce, best_step = dev["ce"], step
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in model.state_dict().items()}
                    save_best(best_state, best_ce, best_step)  # crash-safe
    metrics_file.close()
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    temperature = 1.0
    if data.get("calibration"):
        temperature = fit_temperature(model, data["calibration"], tokenizer.pad_token_id,
                                      device, autocast, args.microbatch_tokens,
                                      per_type=args.per_type_temperature)

    final_metrics = {"best_step": best_step, "dev_ce": best_ce, "temperature": temperature}
    for split in ("test", "ood"):
        if data.get(split):
            final_metrics[f"{split}_ce"] = evaluate_split(
                model, data[split], tokenizer.pad_token_id, device, autocast,
                args.microbatch_tokens)
    track({"final/best_step": best_step, "final/dev_ce": best_ce,
           "final/temperature": temperature}, step)
    if tracker:
        tracker.finish()
    if swan:
        import swanlab as _swanlab

        _swanlab.finish()

    from safetensors.torch import save_file

    save_file({k.replace("._orig_mod.", "."): v for k, v in best_state.items()},
              str(out / "best.safetensors"))
    (out / "config.json").write_text(json.dumps({
        "schema": "jevforge-checkpoint-v1",
        "base_model": str(Path(args.base_model).resolve()),
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "temperature": temperature,
        "best_step": best_step,
        "dev_ce": best_ce,
        "args": vars(args),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copytree(args.base_model, out / "backbone", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pth"))
    (out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(json.dumps(final_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
