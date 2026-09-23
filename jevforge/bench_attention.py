"""Benchmark attention backends on the JevForge training workload shape.

Runs a representative training microbatch (forward+backward with gradient
checkpointing, bf16 autocast) under each requested attn_implementation and
reports peak VRAM and step time. No dataset or generation calls involved.
"""
import argparse
import json
import time

import torch


def run(impl, backbone_dir, batch, length, steps):
    from transformers import AutoConfig, AutoModel

    from .model import JevForgeModel

    config = AutoConfig.from_pretrained(backbone_dir, local_files_only=True,
                                        trust_remote_code=False)
    config.use_cache = False
    body = AutoModel.from_config(config, attn_implementation=impl,
                                 trust_remote_code=False)
    model = JevForgeModel(body, config.hidden_size).cuda()
    model.backbone.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    ids = torch.randint(1000, 50000, (batch, length), device="cuda")
    mask = torch.ones_like(ids)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        start = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(ids, mask)
        (logits.square().mean()).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    result = {"impl": impl, "batch": batch, "length": length,
              "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
              "ms_per_step": round(1000 * sum(times[2:]) / max(len(times) - 2, 1), 1)}
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--impls", default="sdpa,eager")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()
    out = []
    for impl in args.impls.split(","):
        try:
            out.append(run(impl, args.backbone_dir, args.batch, args.length, args.steps))
        except Exception as exc:  # noqa: BLE001 — report and continue with the rest
            out.append({"impl": impl, "error": str(exc)[:120]})
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
