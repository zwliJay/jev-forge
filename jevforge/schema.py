"""JevForge data and wire schemas.

The inference contract mirrors the public TypeSafe "system one" HTTP API
(https://docs.typesafe.ai): one state plus independent questions of type
choice / score / noul. Training records wrap such a request with target
distributions and provenance.
"""
import json
import math

QUESTION_TYPES = ("choice", "score", "noul")
SPLITS = ("train", "dev", "calibration", "test", "ood")
SCHEMA_VERSION = "jevforge-record-v1"


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def validate_question(qid, question):
    if not _nonempty(qid) or not isinstance(question, dict):
        raise ValueError(f"invalid question id or body: {qid!r}")
    extra = set(question) - {"type", "instructions", "criteria"}
    if extra:
        raise ValueError(f"question {qid}: unsupported fields {sorted(extra)}")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"question {qid}: type must be one of {QUESTION_TYPES}")
    if not _nonempty(question.get("instructions")):
        raise ValueError(f"question {qid}: instructions must be a non-empty string")
    criteria = question.get("criteria")
    if kind == "choice":
        if not isinstance(criteria, dict) or not 2 <= len(criteria) <= 255:
            raise ValueError(f"question {qid}: choice criteria must map 2-255 options")
        if not all(_nonempty(k) and _nonempty(v) for k, v in criteria.items()):
            raise ValueError(f"question {qid}: choice ids and descriptions must be non-empty strings")
    elif kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 10:
            raise ValueError(f"question {qid}: score criteria must list 2-10 ordered levels")
        if not all(_nonempty(v) for v in criteria):
            raise ValueError(f"question {qid}: score levels must be non-empty strings")
    else:  # noul
        if "criteria" in question:
            if not isinstance(criteria, dict) or set(criteria) - {"true", "false"}:
                raise ValueError(f"question {qid}: noul criteria may only hold true/false")
            if not all(_nonempty(v) for v in criteria.values()):
                raise ValueError(f"question {qid}: noul criteria must be non-empty strings")
    return question


def candidate_ids(question):
    """Ordered candidate identifiers for each question type."""
    kind = question["type"]
    if kind == "choice":
        return list(question["criteria"])
    if kind == "score":
        return [str(i) for i in range(len(question["criteria"]))]
    return ["false", "true"]


def candidate_texts(question):
    """Text shown on each candidate path (public API semantics)."""
    kind = question["type"]
    if kind == "choice":
        return [f"{k}: {v}" for k, v in question["criteria"].items()]
    if kind == "score":
        return list(question["criteria"])
    return ["The proposition is false.", "The proposition is true."]


def validate_distribution(values, ids, where):
    if not isinstance(values, dict) or set(values) != set(ids):
        raise ValueError(f"{where}: target must cover exactly {ids}")
    numbers = []
    for key in ids:
        value = values[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) \
                or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{where}: probability for {key} must be within [0, 1]")
        numbers.append(float(value))
    if abs(math.fsum(numbers) - 1.0) > 1e-6:
        raise ValueError(f"{where}: probabilities must sum to 1")
    return numbers


def validate_record(record):
    for field in ("id", "split", "source_group"):
        if not _nonempty(record.get(field)):
            raise ValueError(f"record missing {field}")
    if record["split"] not in SPLITS:
        raise ValueError(f"record {record['id']}: bad split {record['split']}")
    request = record.get("request")
    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        raise ValueError(f"record {record['id']}: request must hold state and questions")
    if not _nonempty(request["state"]):
        raise ValueError(f"record {record['id']}: state must be a non-empty string")
    questions = request["questions"]
    if not isinstance(questions, dict) or not questions:
        raise ValueError(f"record {record['id']}: questions must be a non-empty object")
    for qid, question in questions.items():
        validate_question(qid, question)
    targets = record.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError(f"record {record['id']}: targets required")
    kinds = record.get("target_kinds", {})
    for qid, target in targets.items():
        if qid not in questions:
            raise ValueError(f"record {record['id']}: target for unknown question {qid}")
        ids = candidate_ids(questions[qid])
        validate_distribution(target, ids, f"record {record['id']}:{qid}")
        if not _nonempty(kinds.get(qid, "unspecified")):
            raise ValueError(f"record {record['id']}: empty target kind for {qid}")
    return record


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            records.append(validate_record(json.loads(line)))
    return records


def dump_records(records, path):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
