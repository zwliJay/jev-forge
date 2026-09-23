"""Build augmented decision supervision through a configured generation endpoint.

Fills the two supervision gaps the dataset labels cannot cover:
  1. difficulty — a three-level Score target for every record
  2. action augmentation — mixes the train split's sparse policy with a
     generated candidate distribution

The generator is supplied through ``JEVFORGE_GENERATOR_MODEL``. Raw replies are
cached append-only so reruns resume and provenance stays auditable.
"""
import argparse
import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ENDPOINT = os.environ.get("JEVFORGE_CHAT_ENDPOINT", "")
GENERATOR = os.environ.get("JEVFORGE_GENERATOR_MODEL", "")
LEVELS = {"straightforward": "the target element is obvious from the task wording",
          "moderate": "the target requires scanning a few similar elements",
          "intricate": "the target is easy to confuse with several plausible elements"}
DIFFICULTY_QUESTION = {
    "type": "score",
    "instructions": "How difficult is it to identify the correct next element for this task?",
    "criteria": [f"{name}: {desc}" for name, desc in LEVELS.items()],
}


def call_generator(state, elements, api_key, mode, retries=4):
    if mode == "difficulty":
        user = ("Rate how difficult it is for a web agent to pick the correct next "
                f"page element.\nState: {state}\nReply with ONLY a JSON object with "
                "keys \"straightforward\", \"moderate\", \"intricate\" holding "
                "probabilities that sum to 1.")
    else:
        listing = "\n".join(f"{e['id']}: {e['text']}" for e in elements)
        user = ("Score how likely each page element is the correct next interaction "
                f"target.\nState: {state}\nElements:\n{listing}\nReply with ONLY a "
                "JSON object mapping every element id to a score from 0 to 10.")
    body = json.dumps({"model": GENERATOR, "max_tokens": 1600, "temperature": 0.0,
                       "messages": [{"role": "user", "content": user}]}).encode()
    last = None
    for attempt in range(retries):
        request = urllib.request.Request(ENDPOINT, data=body, headers={
            "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.loads(response.read().decode())
            text = payload["choices"][0]["message"]["content"]
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                raise ValueError(f"no JSON in generator reply: {text[:100]}")
            return json.loads(match.group(0)), text
        except Exception as exc:  # noqa: BLE001 — retry transport and parse alike
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"generator failed after {retries} tries: {last}")


def normalize(vector):
    total = math.fsum(vector)
    return None if total <= 0 else [v / total for v in vector]


def assemble_stream(stream_path, output_dir):
    """Snapshot the append-only stream into split files (last record per id wins)."""
    from .schema import SPLITS, dump_records

    seen = {}
    for line in Path(stream_path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except ValueError:  # torn tail line while a writer is mid-append
            continue
        seen[record["id"]] = record
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        dump_records([r for r in seen.values() if r["split"] == split],
                     out / f"{split}.jsonl")
    dump_records(list(seen.values()), out / "all.jsonl")
    return {split: sum(1 for r in seen.values() if r["split"] == split)
            for split in SPLITS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, help="all.jsonl from build_web")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--stream", help="append-only augmented-record stream "
                        "(default <output-dir>/stream.jsonl)")
    parser.add_argument("--api-key-env", default="JEVFORGE_API_KEY")
    parser.add_argument("--soften-lambda", type=float, default=0.25,
                        help="train-split weight for generated candidate probabilities")
    parser.add_argument("--limit", type=int, default=0, help="cap records per split")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--assemble-only", action="store_true",
                        help="skip generation calls; just snapshot the stream")
    args = parser.parse_args()

    stream_path = Path(args.stream or Path(args.output_dir) / "stream.jsonl")
    if args.assemble_only:
        print(json.dumps(assemble_stream(stream_path, args.output_dir)))
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"export {args.api_key_env} first")
    if not ENDPOINT:
        raise SystemExit("export JEVFORGE_CHAT_ENDPOINT first")
    if not GENERATOR:
        raise SystemExit("export JEVFORGE_GENERATOR_MODEL first")

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["id"], row["mode"])] = row

    records = [json.loads(line) for line in
               Path(args.records).read_text(encoding="utf-8").splitlines()]
    by_split = {}
    for record in records:
        by_split.setdefault(record["split"], []).append(record)
    work = []
    for split, items in by_split.items():
        work.extend(items[:args.limit] if args.limit else items)

    def label(record):
        rid = record["id"]
        state = record["request"]["state"]
        elements = json.loads(state)["elements"]
        out = json.loads(json.dumps(record))
        fresh, error = [], None
        try:
            difficulty_row = cache.get((rid, "difficulty"))
            if difficulty_row is None:
                values, raw = call_generator(state, elements, api_key, "difficulty")
                vector = normalize([float(values.get(name, 0.0)) for name in LEVELS])
                if vector is None:
                    return None, [], f"{rid}: bad difficulty reply"
                difficulty_row = {"id": rid, "mode": "difficulty", "generator": GENERATOR,
                                  "raw": raw, "probabilities": vector}
                fresh.append(difficulty_row)
            probs = difficulty_row["probabilities"]
            out["request"]["questions"]["difficulty"] = DIFFICULTY_QUESTION
            out["targets"]["difficulty"] = {str(i): round(p, 6) for i, p in enumerate(probs)}
            out["target_kinds"]["difficulty"] = "generated_score_distribution"

            if args.soften_lambda and record["split"] == "train":
                choice_row = cache.get((rid, "choice"))
                if choice_row is None:
                    values, raw = call_generator(state, elements, api_key, "choice")
                    ids = [e["id"] for e in elements]
                    vector = normalize([float(values.get(eid, 0.0)) for eid in ids])
                    if vector is None:
                        return out, fresh, f"{rid}: bad choice reply (difficulty kept)"
                    choice_row = {"id": rid, "mode": "choice", "generator": GENERATOR,
                                  "raw": raw, "probabilities": vector}
                    fresh.append(choice_row)
                ids = list(record["request"]["questions"]["action"]["criteria"])
                gold = record["targets"]["action"]
                mixed = {eid: (1 - args.soften_lambda) * gold[eid] + args.soften_lambda * t
                         for eid, t in zip(ids, choice_row["probabilities"])}
                total = math.fsum(mixed.values())
                out["targets"]["action"] = {k: v / total for k, v in mixed.items()}
                out["target_kinds"]["action"] = "augmented_soft_policy"
        except Exception as exc:  # noqa: BLE001 — one bad record must not stop the run
            return out, fresh, f"{rid}: {exc}"
        return out, fresh, None

    results, failures = [], []
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool, \
            stream_path.open("a", encoding="utf-8") as stream, \
            cache_path.open("a", encoding="utf-8") as cache_handle:
        futures = [pool.submit(label, record) for record in work]
        for future in as_completed(futures):
            record, fresh, error = future.result()
            done += 1
            if record is None:
                failures.append(error)
                continue
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            for row in fresh:
                cache_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            cache_handle.flush()
            results.append(record)
            if error:
                failures.append(error)
            if done % 50 == 0:
                print(json.dumps({"done": done, "of": len(work),
                                  "failures": len(failures)}), flush=True)

    counts = assemble_stream(stream_path, args.output_dir)
    print(json.dumps({"records": sum(counts.values()), "splits": counts,
                      "failures": len(failures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
