"""Build the mixed Open-Jev / Mind2Web / general training set (8k context).

Productizes the working builders (work/build_candidate_pools_8k.py and
work/build_hybrid_mix_8kctx.py) into one reusable CLI. The JevForge
architecture, prompt encoding and prediction contract are unchanged; this
module only assembles training records in the existing jevforge schema.

Guarantees
----------
* Source split roles only: Bekko ``training-manifest.json`` ``train`` entries
  (row-level ``split == "train"``); Mind2Web source ``train``/``dev``/
  ``calibration`` map to the same output roles. Evaluation/test data never
  enters.
* Authored state, instructions, option descriptions and soft targets are
  preserved verbatim; nothing is relabelled, renormalised or truncated.
  Choice ids render as anonymous ``option_i`` in stored order (matching the
  S1MB anonymous-choice rendering), noul maps by semantic false/true, and
  score accepts only exact ascending ``0..K-1`` scales (the native index
  contract); other numeric scales are rejected with explicit counters.
* Exclusion hashes canonical (key-order independent, whitespace normalised)
  STATE VALUES ONLY — never labels. One canonical state never appears in two
  output splits; full inputs are globally deduplicated; source groups stay
  whole. Overlength or over-budget questions are rejected as complete units
  by the native ``encode_examples`` guard (max_length) plus a padded
  whole-question token budget.
* Mixture quotas are exact (largest-remainder rounding of 60/25/15
  openjev/mind2web/general); sampling is round-robin per task and per source
  config with deterministic shuffling; shortages fail with a clear error.
* No blanket relicensing: upstream terms from the Bekko ``sources.json`` and
  the Mind2Web ``SOURCE.json`` are copied into the manifest. Selected configs
  and their license fitness are the caller's responsibility.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path

from .encode import encode_examples
from .schema import dump_records, validate_record

RECIPE = "mixed-8kctx-60pct-openjev-25pct-mind2web-15pct-general-v1"
BEKKO_REPO = "hotchpotch/bekko-system-one-dataset-v0"
MIND2WEB_REPO = "AndeyTait/JevForge-Mind2Web"
MIXTURE_RATIOS = (("openjev", 0.60), ("mind2web", 0.25), ("general", 0.15))
KINDS = ("choice", "noul", "score")
ROLES = ("train", "dev", "calibration")
DEFAULT_LANGUAGES = ("en", "eng", "english", "und", None)
SHA40 = re.compile(r"^[0-9a-f]{40}$")


class MapRejected(Exception):
    """A decision cannot be mapped; ``reason`` feeds the rejection counters."""

    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


# --- canonical hashing ------------------------------------------------------


def _clean(value):
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, dict):
        return {key: _clean(value[key]) for key in value}
    if isinstance(value, list):
        return [_clean(item) for item in value]
    return value


def canonical_state(state):
    """Key-order independent, whitespace-normalised canonical state text."""
    value = state
    if isinstance(state, str):
        try:
            value = json.loads(state)
        except (ValueError, TypeError):
            value = state
    return json.dumps(_clean(value), sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def state_hash(state):
    return hashlib.sha256(canonical_state(state).encode("utf-8")).hexdigest()


def sha256_file(path):
    """Streaming file digest; snapshots can be large."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_identity(state, question):
    payload = json.dumps({"state": _clean(_parse(state)), "question": question},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(" ".join(payload.split()).encode("utf-8")).hexdigest()


def _parse(state):
    if isinstance(state, str):
        try:
            return json.loads(state)
        except (ValueError, TypeError):
            return state
    return state


def partition_for_group(group, seed=42):
    """Stable 80/10/10 group partition (train/dev/calibration)."""
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8],
                 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "calibration"


def padded_question_cost(encoded_example):
    """Whole padded question cost: padded width times candidate count."""
    width = (max(len(leaf) for leaf in encoded_example["leaves"]) + 63) // 64 * 64
    return width * len(encoded_example["leaves"])


# --- mixture quotas ---------------------------------------------------------


def compute_mixture_quotas(total, ratios=MIXTURE_RATIOS):
    """Exact largest-remainder split of ``total`` over the mixture ratios."""
    raw = {name: total * share for name, share in ratios}
    quotas = {name: int(math.floor(value)) for name, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    assert sum(quotas.values()) == total
    return quotas


def task_quotas(total, sizes):
    """Split one source quota over task kinds, mirroring the hybrid builder."""
    quotas = {kind: min(total // len(KINDS), sizes.get(kind, 0)) for kind in KINDS}
    remaining = total - sum(quotas.values())
    progressed = True
    while remaining and progressed:
        progressed = False
        for kind in KINDS:
            if remaining and quotas[kind] < sizes.get(kind, 0):
                quotas[kind] += 1
                remaining -= 1
                progressed = True
    return quotas, remaining


# --- Bekko decision mapping -------------------------------------------------


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _loads_or_reject(text, reason="invalid_json"):
    try:
        return json.loads(text)
    except (ValueError, TypeError) as exc:
        raise MapRejected(reason) from exc


def map_bekko_decision(decision, target, max_candidates=32):
    """Map one Bekko judgment decision to a jevforge question + target."""
    if decision.get("kind") != "judgment":
        raise MapRejected("unsupported_kind")
    if decision.get("documents"):
        raise MapRejected("ranking_documents")
    if decision.get("scoring") is not None:
        raise MapRejected("relative_scoring")
    kind = decision.get("type")
    options = list(decision.get("criteria") or [])
    ids = [option.get("id") for option in options]
    if len(set(ids)) != len(ids) or not 2 <= len(ids) <= max_candidates:
        raise MapRejected("candidate_count")
    if target is None:
        raise MapRejected("missing_target")
    if target.get("kind") != "judgment_distribution":
        raise MapRejected("unsupported_target_kind")
    target_ids = list(target.get("ids") or [])
    probabilities = list(target.get("probabilities") or [])
    if len(target_ids) != len(probabilities) or len(set(target_ids)) != len(target_ids):
        raise MapRejected("target_length_mismatch")
    numbers = []
    for value in probabilities:
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise MapRejected("invalid_target_probabilities")
        numbers.append(float(value))
    if abs(math.fsum(numbers) - 1.0) > 1e-6:
        raise MapRejected("invalid_target_probabilities")
    probs = dict(zip(target_ids, numbers))
    if set(probs) != set(ids):
        raise MapRejected("target_coverage")

    instructions = _text(_loads_or_reject(decision.get("instructions_json")))
    system = decision.get("system_prompt") or ""
    instructions = "\n\n".join(part for part in (system, instructions) if part)
    descriptions = [_text(_loads_or_reject(option.get("description_json")))
                    for option in options]

    if kind == "choice":
        criteria = {f"option_{i}": text for i, text in enumerate(descriptions)}
        new_target = {f"option_{i}": probs[option["id"]]
                      for i, option in enumerate(options)}
    elif kind == "noul":
        if set(ids) != {"false", "true"}:
            raise MapRejected("noul_ids")
        criteria = {option["id"]: text
                    for option, text in zip(options, descriptions)}
        new_target = {key: probs[key] for key in ("false", "true")}
    elif kind == "score":
        if len(options) > 10:
            raise MapRejected("score_levels")
        # Native contract: score candidates are indices, so only exact
        # ascending 0..K-1 numeric scales preserve the authored meaning.
        if [option.get("value") for option in options] != list(range(len(options))):
            raise MapRejected("nonindex_score_scale")
        criteria = descriptions
        new_target = {str(i): probs[option["id"]]
                      for i, option in enumerate(options)}
    else:
        raise MapRejected("unknown_type")
    question = {"type": kind, "instructions": instructions, "criteria": criteria}
    return question, new_target


def map_bekko_row(row, config, revision, max_candidates=32,
                  languages=DEFAULT_LANGUAGES, seed=42,
                  dataset_repo=BEKKO_REPO):
    """Map one Bekko row to per-decision records plus rejection counters.

    Returns ``(records, counters)``; acceptance guards (exclusion, dedup,
    split isolation, length budgets) are applied later by :func:`make_guard`.
    """
    counters = collections.Counter()
    if row.get("split") != "train":
        counters["not_train"] += 1
        return [], counters
    if row.get("language") not in list(languages):
        counters["not_english"] += 1
        return [], counters
    source = row.get("input") or {}
    try:
        state = _text(json.loads(source["state_json"]))
    except (KeyError, ValueError, TypeError):
        counters["rejected:invalid_state"] += 1
        return [], counters
    canonical = state_hash(state)
    group = row.get("group_id") or canonical
    split = partition_for_group(group, seed)
    targets = {entry.get("decision_id"): entry
               for entry in row.get("targets") or []
               if entry.get("kind") == "judgment_distribution"}
    records = []
    for decision in source.get("decisions") or []:
        qid = decision.get("id")
        if not isinstance(qid, str) or not qid.strip():
            counters["rejected:missing_question_id"] += 1
            continue
        try:
            question, target = map_bekko_decision(
                decision, targets.get(qid), max_candidates)
        except MapRejected as exc:
            counters["rejected:" + exc.reason] += 1
            continue
        records.append({
            "id": f"{config}:{row.get('case_id')}:{qid}",
            "schema_version": "jevforge-record-v1",
            "split": split,
            "source_group": group,
            "request": {"state": state, "questions": {qid: question}},
            "targets": {qid: target},
            "target_kinds": {qid: (targets[qid].get("annotation_kind")
                                   or "unspecified")},
            "provenance": {
                "dataset": dataset_repo, "revision": revision, "config": config,
                "case_id": row.get("case_id"),
                "input_hash": row.get("input_hash"),
                "canonical_state_hash": canonical,
            },
        })
    return records, counters


# --- Mind2Web mapping -------------------------------------------------------


def map_mind2web_record(record, revision, dataset_repo=MIND2WEB_REPO):
    """Split one Mind2Web schema record into per-decision records.

    The source split role is preserved exactly (train/dev/calibration);
    other roles are skipped and counted so they can never train or calibrate.
    """
    counters = collections.Counter()
    split = record.get("split")
    if split not in ROLES:
        counters["mind2web_role_skipped"] += 1
        return [], counters
    records = []
    kinds = record.get("target_kinds", {})
    for qid, question in record["request"]["questions"].items():
        records.append({
            "id": f"{record['id']}:{qid}",
            "schema_version": "jevforge-record-v1",
            "split": split,
            "source_group": f"mind2web:{record['source_group']}",
            "request": {"state": record["request"]["state"],
                        "questions": {qid: question}},
            "targets": {qid: record["targets"][qid]},
            "target_kinds": {qid: kinds.get(qid, "unspecified")},
            "provenance": {
                "dataset": dataset_repo, "revision": revision,
                "config": "website-disjoint", "case_id": record["id"],
                "original_split": split,
                "original_source_group": record["source_group"],
            },
        })
    return records, counters


# --- acceptance guard -------------------------------------------------------


def make_guard(tokenizer, blocked_hashes, max_length=8192,
               question_token_budget=16384):
    """Shared acceptance guard: exclusion, split isolation, dedup, budgets."""
    seen_inputs = set()
    state_split = {}
    counters = collections.Counter()

    def accept(record, mixture_source):
        try:
            validate_record(record)
        except (ValueError, TypeError, KeyError) as exc:
            counters["rejected:schema:" + type(exc).__name__] += 1
            return False
        canonical = state_hash(record["request"]["state"])
        if canonical in blocked_hashes:
            counters["canonical_evaluation_state_overlap"] += 1
            return False
        split = record["split"]
        if canonical in state_split and state_split[canonical] != split:
            counters["cross_split_state"] += 1
            return False
        state_split[canonical] = split
        (qid, question), = record["request"]["questions"].items()
        identity = input_identity(record["request"]["state"], question)
        if identity in seen_inputs:
            counters["duplicate_input"] += 1
            return False
        try:
            encoded = encode_examples(record["request"], tokenizer, max_length)
            for example in encoded:
                if padded_question_cost(example) > question_token_budget:
                    raise ValueError("question token budget")
        except ValueError:
            counters["overlength_whole_question"] += 1
            return False
        seen_inputs.add(identity)
        record["provenance"]["canonical_state_hash"] = canonical
        record["provenance"]["mixture_source"] = mixture_source
        return True

    return accept, counters


# --- exclusion file ---------------------------------------------------------


def load_exclude_states(path):
    """Read a JSONL of canonical or raw STATE VALUES ONLY.

    Each line may be a bare JSON value (string or structured state) or a
    wrapper ``{"state"|"state_json"|"canonical_state": <value>,
    "provenance": {...}}``. Labels or targets are never read. Returns
    ``(hashes, provenance, file_sha256)``.
    """
    path = Path(path)
    hashes, provenance = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        value = json.loads(line)
        if isinstance(value, dict) and "provenance" in value:
            wrapper = value
            for key in ("state", "state_json", "canonical_state"):
                if key in wrapper:
                    value = wrapper[key]
                    provenance.append(wrapper.get("provenance"))
                    break
            else:
                continue
        hashes.add(state_hash(value))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes, [item for item in provenance if item is not None], digest


# --- Bekko parquet scanning -------------------------------------------------


def iter_bekko_rows(bekko_dir, data_files, scan_rows=5000, seed=42):
    """Deterministically sample capped row groups, never scanning everything."""
    import pyarrow.parquet as pq  # lazy: tests without pyarrow still import

    root = Path(bekko_dir)
    emitted = 0
    for relative in data_files:
        parquet = pq.ParquetFile(root / relative)
        groups = list(range(parquet.num_row_groups))
        random.Random(seed).shuffle(groups)
        for group_index in groups:
            for row in parquet.read_row_group(group_index).to_pylist():
                yield row
                emitted += 1
                if emitted >= scan_rows:
                    return
            if emitted >= scan_rows:
                return


def cap_for(config, cap_openjev, cap_general, cap_configs):
    if config in cap_configs:
        return int(cap_configs[config])
    if config.startswith("open_jev__"):
        return cap_openjev
    return cap_general


# --- quota selection --------------------------------------------------------


def select_quotas(pool, split, source, total, seed=42, oversample=1):
    """Round-robin per-task/per-config sampling with exact quotas.

    ``oversample > 1`` (train split only) cyclically repeats already selected
    records to reach the quota; copies are counted and reported.
    """
    queues = {kind: [list(records) for (split_, source_, kind_, _config), records
                     in sorted(pool.items())
                     if split_ == split and source_ == source and kind_ == kind]
              for kind in KINDS}
    sizes = {kind: sum(len(queue) for queue in queues[kind]) for kind in KINDS}
    quotas, remaining = task_quotas(total, sizes)
    if remaining and not (oversample > 1 and split == "train"):
        raise RuntimeError(
            f"quota shortage: {split}/{source} needs {total} decisions but the "
            f"pools hold {sum(sizes.values())} (per task {sizes}); raise scan "
            f"rows, per-config caps or adjust quotas")
    if remaining:  # train oversampling: take everything, then repeat below
        quotas = dict(sizes)
    picked = []
    copies = 0
    for kind in KINDS:
        rng = random.Random(seed)
        rng.shuffle(queues[kind])
        for queue in queues[kind]:
            rng.shuffle(queue)
        count = 0
        while count < quotas[kind]:
            for queue in queues[kind]:
                if queue and count < quotas[kind]:
                    picked.append(queue.pop())
                    count += 1
    if len(picked) < total and oversample > 1:
        base = list(picked)
        index = 0
        while len(picked) < total:
            picked.append(base[index % len(base)])
            copies += 1
            index += 1
    return picked, {"selected": len(picked), "oversampled_copies": copies,
                    "task_quotas": quotas}


# --- top-level assembly -----------------------------------------------------


def assemble_outputs(pool, quotas, seed=42, oversample_by_source=None):
    oversample_by_source = oversample_by_source or {}
    outputs, selection = {}, {}
    for split, source_totals in quotas.items():
        records = []
        selection[split] = {}
        for source, total in source_totals.items():
            oversample = oversample_by_source.get(source, 1) \
                if split == "train" else 1
            chosen, info = select_quotas(pool, split, source, total, seed,
                                         oversample)
            records.extend(chosen)
            selection[split][source] = info
        random.Random(seed).shuffle(records)
        outputs[split] = records
    for first, second in (("train", "dev"), ("train", "calibration"),
                          ("dev", "calibration")):
        for key in ("source_group",):
            overlap = {record[key] for record in outputs[first]} & \
                      {record[key] for record in outputs[second]}
            if overlap:
                raise RuntimeError(f"group leak {first}/{second}: {sorted(overlap)[:3]}")
        overlap = {record["provenance"]["canonical_state_hash"]
                   for record in outputs[first]} & \
                  {record["provenance"]["canonical_state_hash"]
                   for record in outputs[second]}
        if overlap:
            raise RuntimeError(f"state leak {first}/{second}")
    return outputs, selection


def write_outputs(outputs, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, records in outputs.items():
        path = output / f"{split}.jsonl"
        dump_records(records, path)  # validates every record again
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def split_stats(records):
    return {
        "decisions": len(records),
        "mixture_sources": dict(collections.Counter(
            record["provenance"]["mixture_source"] for record in records)),
        "tasks": dict(collections.Counter(
            next(iter(record["request"]["questions"].values()))["type"]
            for record in records)),
        "sources": dict(collections.Counter(
            record["provenance"]["config"] for record in records)),
    }


def build(bekko_dir, bekko_revision, mind2web_dir, mind2web_revision,
          configs, tokenizer, output_dir, train_decisions=16000,
          dev_decisions=300, calibration_decisions=150, max_length=8192,
          max_candidates=32, question_token_budget=16384, seed=42,
          exclude_states=None, cap_openjev=1200, cap_general=80,
          cap_configs=None, openjev_oversample=1, scan_rows=5000,
          languages=DEFAULT_LANGUAGES):
    """Run the full build; returns the provenance manifest."""
    cap_configs = cap_configs or {}
    bekko_dir, mind2web_dir = Path(bekko_dir), Path(mind2web_dir)
    training_manifest = json.loads(
        (bekko_dir / "training-manifest.json").read_text(encoding="utf-8"))
    sources_json = json.loads(
        (bekko_dir / "sources.json").read_text(encoding="utf-8"))
    selected = set(configs)

    blocked, exclusion_provenance, exclusion_sha = set(), [], None
    if exclude_states:
        blocked, exclusion_provenance, exclusion_sha = load_exclude_states(
            exclude_states)

    accept, counters = make_guard(tokenizer, blocked, max_length,
                                  question_token_budget)
    pool = collections.defaultdict(list)
    file_hashes, declared = {}, {}

    for entry in training_manifest.get("train", []):
        config = entry.get("dataset")
        if config not in selected:
            continue
        accepted = 0
        cap = cap_for(config, cap_openjev, cap_general, cap_configs)
        for relative in entry.get("data_files", []):
            path = bekko_dir / relative
            if str(path) not in file_hashes:
                file_hashes[str(path)] = sha256_file(path)
        for row in iter_bekko_rows(bekko_dir, entry.get("data_files", []),
                                   scan_rows, seed):
            if accepted >= cap:
                break
            records, row_counters = map_bekko_row(
                row, config, bekko_revision, max_candidates, languages, seed)
            counters.update(row_counters)
            source = "openjev" if config.startswith("open_jev__") else "general"
            for record in records:
                if accepted >= cap:
                    counters["cap_truncated"] += 1
                    break
                if accept(record, source):
                    pool[record["split"], source,
                         next(iter(record["request"]["questions"]
                                   .values()))["type"], config].append(record)
                    accepted += 1
        declared[config] = {"accepted": accepted, "cap": cap}

    mind2web_files = {}
    source_json = json.loads(
        (mind2web_dir / "SOURCE.json").read_text(encoding="utf-8"))
    for split in ROLES:
        path = mind2web_dir / "data" / f"{split}.jsonl"
        mind2web_files[str(path)] = sha256_file(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            records, role_counters = map_mind2web_record(
                json.loads(line), mind2web_revision)
            counters.update(role_counters)
            for record in records:
                if accept(record, "mind2web"):
                    pool[record["split"], "mind2web",
                         next(iter(record["request"]["questions"]
                                   .values()))["type"],
                         record["provenance"]["config"]].append(record)
    checksums = mind2web_dir / "checksums.sha256"
    if checksums.exists():
        for line in checksums.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2:
                digest, relative = parts
                path = mind2web_dir / relative.lstrip("*")
                if path.is_file() and sha256_file(path) != digest:
                    raise RuntimeError(f"mind2web checksum mismatch: {path}")
                mind2web_files[str(path)] = digest

    totals = {"train": train_decisions, "dev": dev_decisions,
              "calibration": calibration_decisions}
    quotas = {split: compute_mixture_quotas(total) for split, total in totals.items()}
    outputs, selection = assemble_outputs(
        pool, quotas, seed, {"openjev": openjev_oversample})
    output_hashes = write_outputs(outputs, output_dir)

    terms = {}
    for config in sorted({record["provenance"]["config"]
                          for records in outputs.values()
                          for record in records
                          if record["provenance"]["mixture_source"] != "mind2web"}):
        terms[config] = (sources_json.get("datasets", {}).get(config, {})
                         .get("train_and_validation", {})
                         .get("upstream_datasets", []))
    manifest = {
        "recipe": RECIPE,
        "seed": seed,
        "architecture_changed": False,
        "max_length": max_length,
        "max_candidates": max_candidates,
        "question_token_budget": question_token_budget,
        "score_policy": "Bekko retains only exact ascending 0..K-1 score "
                        "scales (native index contract); other numeric scales "
                        "are rejected whole and counted, never remapped",
        "mixture_ratios": dict(MIXTURE_RATIOS),
        "quotas": {split: dict(quotas[split]) for split in quotas},
        "selection": selection,
        "source_revisions": {
            "bekko": {"repo_id": BEKKO_REPO, "revision": bekko_revision},
            "mind2web": {"repo_id": MIND2WEB_REPO, "revision": mind2web_revision},
        },
        "source_input_files_sha256": {**file_hashes, **mind2web_files},
        "bekko_train_role_only": True,
        "bekko_configs_declared": declared,
        "scan_rows_per_config": scan_rows,
        "evaluation_input_exclusion": {
            "exclude_states_file": str(exclude_states) if exclude_states else None,
            "file_sha256": exclusion_sha,
            "canonical_state_hashes": len(blocked),
            "declared_provenance": exclusion_provenance,
            "labels_or_scores_used": False,
        },
        "rejections": dict(counters),
        "splits": {split: {**split_stats(outputs[split]),
                           "sha256": output_hashes[split]}
                   for split in outputs},
        "source_terms": terms,
        "mind2web_upstream": source_json,
        "license_note": sources_json.get("note"),
        "disclaimers": [
            "No blanket relicensing: upstream terms are recorded per config.",
            "Selected configs and their license fitness are the caller's "
            "responsibility; this build asserts no license conclusions.",
        ],
    }
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the mixed Open-Jev/Mind2Web/general training set.")
    parser.add_argument("--bekko-dir", required=True,
                        help="local pinned Bekko snapshot directory")
    parser.add_argument("--bekko-revision", required=True,
                        help="exact 40-hex Bekko commit SHA")
    parser.add_argument("--mind2web-dir", required=True,
                        help="local Mind2Web release (data/*.jsonl, SOURCE.json)")
    parser.add_argument("--mind2web-revision", required=True,
                        help="exact 40-hex Mind2Web commit SHA")
    parser.add_argument("--configs-json", required=True,
                        help="JSON file listing selected Bekko config names")
    parser.add_argument("--tokenizer", required=True,
                        help="local tokenizer/model path for the encode guard")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-decisions", type=int, default=16000)
    parser.add_argument("--dev-decisions", type=int, default=300)
    parser.add_argument("--calibration-decisions", type=int, default=150)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--question-token-budget", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-states", default=None,
                        help="JSONL of canonical/raw state values (no labels)")
    parser.add_argument("--cap-openjev", type=int, default=1200)
    parser.add_argument("--cap-general", type=int, default=80)
    parser.add_argument("--cap-configs", default=None,
                        help="JSON file mapping config name to acceptance cap")
    parser.add_argument("--openjev-oversample", type=int, default=1,
                        help=">1 repeats openjev train picks to reach quota")
    parser.add_argument("--scan-rows", type=int, default=5000)
    parser.add_argument("--languages", default="en,eng,english,und",
                        help="comma-separated language whitelist")
    args = parser.parse_args(argv)
    for name in ("bekko_revision", "mind2web_revision"):
        value = getattr(args, name)
        if not SHA40.match(value):
            parser.error(f"--{name.replace('_', '-')} must be a 40-hex SHA")
    configs = json.loads(Path(args.configs_json).read_text(encoding="utf-8"))
    cap_configs = json.loads(Path(args.cap_configs).read_text(
        encoding="utf-8")) if args.cap_configs else {}
    languages = tuple({None if token == "None" else token
                       for token in args.languages.split(",")})
    from transformers import AutoTokenizer  # lazy: tests use a fake tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    manifest = build(
        bekko_dir=args.bekko_dir, bekko_revision=args.bekko_revision,
        mind2web_dir=args.mind2web_dir, mind2web_revision=args.mind2web_revision,
        configs=configs, tokenizer=tokenizer, output_dir=args.output_dir,
        train_decisions=args.train_decisions,
        dev_decisions=args.dev_decisions,
        calibration_decisions=args.calibration_decisions,
        max_length=args.max_length, max_candidates=args.max_candidates,
        question_token_budget=args.question_token_budget, seed=args.seed,
        exclude_states=args.exclude_states, cap_openjev=args.cap_openjev,
        cap_general=args.cap_general, cap_configs=cap_configs,
        openjev_oversample=args.openjev_oversample, scan_rows=args.scan_rows,
        languages=languages)
    print(json.dumps({split: stats["decisions"]
                      for split, stats in manifest["splits"].items()}))


if __name__ == "__main__":
    main()
