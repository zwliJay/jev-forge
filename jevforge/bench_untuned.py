"""Untuned-backbone baselines on the fixed held-out choice questions.

Arms:
  local — the raw Qwen3-0.6B base model (the same backbone JevForge fine-tunes),
          answering via a chat prompt on the local GPU; nothing trained.
  api   — a reference chat model through a configured API endpoint.

Both score every candidate element 0-10 and are normalised into a
distribution, mirroring the generated-reference protocol used for the comparison arm.
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHAT_ENDPOINT = os.environ.get("JEVFORGE_CHAT_ENDPOINT", "")


def scoring_prompt(state, question):
    listing = "\n".join(f"{k}: {v}" for k, v in question["criteria"].items())
    return ("Score how likely each page element is the correct next interaction "
            f"target.\nState: {state}\nElements:\n{listing}\n"
            "Reply with ONLY a JSON object mapping every element id to a score from 0 to 10.")


def parse_scores(text, ids):
    match = re.search(r"\{.*\}", text, re.S)
    values = json.loads(match.group(0)) if match else {}
    total = math.fsum(float(values.get(i, 0.0)) for i in ids) or 1.0
    return {i: float(values.get(i, 0.0)) / total for i in ids}


def score_choice(record, probabilities):
    target = record["targets"]["action"]
    ids = list(record["request"]["questions"]["action"]["criteria"])
    probs = [probabilities.get(i, 0.0) for i in ids]
    positives = {i for i in ids if target[i] > 0}
    top = max(range(len(ids)), key=probs.__getitem__)
    return {"top1_in_positives": ids[top] in positives,
            "exact_gold": ids[top] == record["gold"]["action"],
            "mass_on_positives": math.fsum(p for i, p in zip(ids, probs) if i in positives)}


def run_local(records, base_dir, device="cuda:0", batch=16):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_dir, local_files_only=True,
                                              trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        trust_remote_code=False).to(device).eval()
    rows, times, start_all = [], [], time.perf_counter()
    for offset in range(0, len(records), batch):
        chunk = records[offset:offset + batch]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user",
              "content": scoring_prompt(r["request"]["state"],
                                        r["request"]["questions"]["action"])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in chunk]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           padding_side="left").to(device)
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        times.append(time.perf_counter() - start)
        for i, record in enumerate(chunk):
            reply = tokenizer.decode(out[i][inputs.input_ids.shape[1]:],
                                     skip_special_tokens=True)
            ids = list(record["request"]["questions"]["action"]["criteria"])
            rows.append(score_choice(record, parse_scores(reply, ids)))
    return rows, {
        "mean_batch_latency_s": round(math.fsum(times) / max(len(times), 1), 3),
        "records_per_s": round(len(records) / (time.perf_counter() - start_all), 2),
    }


def run_api(records, model, api_key, concurrency=8):
    def one(record):
        question = record["request"]["questions"]["action"]
        body = json.dumps({"model": model, "max_tokens": 1600, "temperature": 0.0,
                           "messages": [{"role": "user", "content":
                                         scoring_prompt(record["request"]["state"], question)}]}).encode()
        request = urllib.request.Request(CHAT_ENDPOINT, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            text = json.loads(response.read().decode())["choices"][0]["message"]["content"]
        return score_choice(record, parse_scores(text, list(question["criteria"])))

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, records)), {}


def mean(rows, key):
    return round(math.fsum(r[key] for r in rows) / max(len(rows), 1), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="dataset dir")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--arm", choices=["local", "api"], required=True)
    parser.add_argument("--local-model", help="local Qwen3-0.6B snapshot dir")
    parser.add_argument("--api-model", default="qwen/qwen3.5-flash-02-23")
    parser.add_argument("--api-key-env", default="JEVFORGE_API_KEY")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = {"arm": args.arm, "model": args.local_model if args.arm == "local"
              else args.api_model, "splits": {}}
    for split in args.splits.split(","):
        records = [json.loads(line) for line in
                   (Path(args.records) / f"{split}.jsonl").read_text(
                       encoding="utf-8").splitlines()][:args.limit]
        if args.arm == "local":
            rows, extra = run_local(records, args.local_model)
        else:
            rows, extra = run_api(records, args.api_model,
                                  os.environ.get(args.api_key_env), args.concurrency)
        report["splits"][split] = {
            "questions": len(rows),
            "choice": {k: mean(rows, k) for k in
                       ("top1_in_positives", "exact_gold", "mass_on_positives")},
            **extra,
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

