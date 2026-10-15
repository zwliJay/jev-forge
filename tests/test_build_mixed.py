"""Tests for jevforge.build_mixed — the mixed Open-Jev/Mind2Web/general builder.

These exercise the ACTUAL imported production functions (no mirrored
implementation). Fixtures are minimal synthetic equivalents of the saved real
Bekko rows (work/samples/bekko-rows.json, kept outside the repo) and of
Mind2Web release records. A fake tokenizer avoids torch/model downloads;
pyarrow is only needed for the parquet-backed tests.
"""
import collections
import json
import shutil

import pytest

from jevforge.build_mixed import (
    DEFAULT_LANGUAGES,
    SHA40,
    build,
    canonical_state,
    compute_mixture_quotas,
    input_identity,
    iter_bekko_rows,
    load_exclude_states,
    make_guard,
    map_bekko_row,
    map_mind2web_record,
    padded_question_cost,
    partition_for_group,
    select_quotas,
    state_hash,
)
from jevforge.schema import load_records


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(range(max(1, len(text.split()))))


# --- fixtures ---------------------------------------------------------------


def decision(did="q", kind="choice", ids=("a", "b"), descriptions=("A", "B"),
             instructions="Pick.", system="", values=None, kind_field="judgment",
             documents=None):
    if values is None:
        values = [None] * len(ids)
    return {
        "id": did, "kind": kind_field, "type": kind,
        "instructions_json": json.dumps(instructions),
        "system_prompt": system,
        "criteria": [{"id": i, "description_json": json.dumps(d), "value": v}
                     for i, d, v in zip(ids, descriptions, values)],
        "documents": documents or [], "scoring": None,
    }


def target(did="q", ids=("a", "b"), probabilities=(1.0, 0.0),
           annotation="hard_label"):
    return {"decision_id": did, "kind": "judgment_distribution",
            "annotation_kind": annotation, "ids": list(ids),
            "probabilities": list(probabilities), "metadata_json": "{}"}


def bekko_row(case_id="c1", group="g1", split="train", language="en",
              decisions=(), targets=(), state=None, input_hash="h"):
    state = state if state is not None else {"scene": "shared situation"}
    return {"input": {"state_json": json.dumps(state, ensure_ascii=False),
                      "decisions": list(decisions)},
            "targets": list(targets), "split": split, "case_id": case_id,
            "group_id": group, "language": language, "input_hash": input_hash}


def find_group(prefix, wanted):
    index = 0
    while True:
        group = f"{prefix}{index}"
        if partition_for_group(group) == wanted:
            return group
        index += 1


# --- canonical hashing and partitions ---------------------------------------


def test_canonical_state_is_key_order_and_whitespace_independent():
    raw = '{"b": "x   y", "a": {"z": 1}}'
    reordered = '{"a":{"z":1},"b":"x y"}'
    assert canonical_state(raw) == canonical_state(reordered)
    assert canonical_state({"b": "x y", "a": {"z": 1}}) == canonical_state(raw)
    assert canonical_state("plain text") == '"plain text"'
    assert state_hash(raw) == state_hash(reordered)


def test_input_identity_dedups_key_order_variants():
    question = {"type": "noul", "instructions": "P.", "criteria": {}}
    assert input_identity('{"a":1,"b":2}', question) == \
        input_identity('{"b": 2, "a": 1}', question)


def test_partition_for_group_is_stable_and_bounded():
    assert partition_for_group("g") == partition_for_group("g")
    assert {partition_for_group(f"g{i}") for i in range(300)} <= \
        {"train", "dev", "calibration"}
    assert find_group("t", "train") and find_group("d", "dev") \
        and find_group("c", "calibration")


def test_mixture_quotas_exact_rounding():
    assert compute_mixture_quotas(16000) == \
        {"openjev": 9600, "mind2web": 4000, "general": 2400}
    for total in range(0, 40):
        quotas = compute_mixture_quotas(total)
        assert sum(quotas.values()) == total
        assert all(value >= 0 for value in quotas.values())


# --- Bekko mapping semantics -------------------------------------------------


def test_soft_targets_follow_ids_not_position_under_candidate_reorder():
    row = bekko_row(
        decisions=[decision("q", "choice", ("a", "b"), ("A text", "B text"))],
        targets=[target("q", ("b", "a"), (0.7, 0.3))])
    records, counters = map_bekko_row(row, "cfg", "r" * 40)
    assert not counters
    assert records[0]["targets"]["q"] == {"option_0": 0.3, "option_1": 0.7}
    assert list(records[0]["request"]["questions"]["q"]["criteria"]) == \
        ["option_0", "option_1"]
    assert list(records[0]["request"]["questions"]["q"]["criteria"].values()) == \
        ["A text", "B text"]


def test_noul_semantic_false_true_mapping_with_reversed_storage():
    row = bekko_row(
        decisions=[decision("n1", "noul", ("true", "false"), ("T def", "F def"),
                            instructions="Proposition.")],
        targets=[target("n1", ("true", "false"), (0.7, 0.3),
                        annotation="teacher_distribution")])
    records, counters = map_bekko_row(row, "cfg", "r" * 40)
    assert not counters
    question = records[0]["request"]["questions"]["n1"]
    assert question["criteria"] == {"true": "T def", "false": "F def"}
    assert records[0]["targets"]["n1"] == {"false": 0.3, "true": 0.7}
    assert records[0]["target_kinds"]["n1"] == "teacher_distribution"


def test_score_accepts_index_scales_rejects_other_numeric_scales():
    ok = bekko_row(
        decisions=[decision("s", "score", ("0", "1", "2"), ("lo", "mid", "hi"),
                            values=(0.0, 1.0, 2.0))],
        targets=[target("s", ("0", "1", "2"), (0.2, 0.3, 0.5))])
    records, counters = map_bekko_row(ok, "cfg", "r" * 40)
    assert records[0]["targets"]["s"] == {"0": 0.2, "1": 0.3, "2": 0.5}
    assert counters == collections.Counter()

    nonindex = bekko_row(
        decisions=[decision("s", "score", ("0", "1"), ("lo", "hi"),
                            values=(1.0, 2.5))],
        targets=[target("s", ("0", "1"), (0.5, 0.5))])
    records, counters = map_bekko_row(nonindex, "cfg", "r" * 40)
    assert records == []
    assert counters["rejected:nonindex_score_scale"] == 1


def test_structured_instructions_and_system_prompt_join():
    row = bekko_row(
        decisions=[{"id": "v", "kind": "judgment", "type": "score",
                    "instructions_json": json.dumps(
                        {"task": "Judge length.", "question": "How long?"}),
                    "system_prompt": "SYS",
                    "criteria": [{"id": "0", "value": 0,
                                  "description_json": json.dumps("short")},
                                 {"id": "1", "value": 1,
                                  "description_json": json.dumps("long")}],
                    "documents": [], "scoring": None}],
        targets=[target("v", ("0", "1"), (0.5, 0.5))])
    records, _ = map_bekko_row(row, "cfg", "r" * 40)
    instructions = records[0]["request"]["questions"]["v"]["instructions"]
    assert instructions == 'SYS\n\n{"task": "Judge length.", "question": "How long?"}'
    assert json.loads(instructions.split("\n\n", 1)[1])["task"] == "Judge length."


def test_bekko_row_level_rejections_are_explicit():
    ids33 = tuple(f"o{i}" for i in range(33))
    cases = [
        (bekko_row(decisions=[decision("q", "choice", ids33, ids33)],
                   targets=[target("q", ids33, [1.0] + [0.0] * 32)]),
         "rejected:candidate_count"),
        (bekko_row(decisions=[decision("q")], targets=[]), "rejected:missing_target"),
        (bekko_row(decisions=[decision("q", kind_field="ranking",
                                       documents=[{"id": "d"}])],
                   targets=[target("q")]), "rejected:unsupported_kind"),
        (bekko_row(split="validation"), "not_train"),
        (bekko_row(language="ja"), "not_english"),
    ]
    for row, reason in cases:
        records, counters = map_bekko_row(row, "cfg", "r" * 40)
        assert records == [], reason
        assert counters[reason] == 1, reason


def test_score_above_ten_levels_and_oversize_targets_rejected():
    ids = tuple(str(i) for i in range(11))
    row = bekko_row(
        decisions=[decision("s", "score", ids, ids, values=tuple(range(11)))],
        targets=[target("s", ids, [1.0] + [0.0] * 10)])
    records, counters = map_bekko_row(row, "cfg", "r" * 40)
    assert counters["rejected:score_levels"] == 1
    bad_sum = bekko_row(
        decisions=[decision("q", "choice", ("a", "b"), ("A", "B"))],
        targets=[target("q", ("a", "b"), (0.9, 0.9))])
    records, counters = map_bekko_row(bad_sum, "cfg", "r" * 40)
    assert counters["rejected:invalid_target_probabilities"] == 1


def test_state_preserved_verbatim_as_parsed_dumps():
    state = {"b": 1, "a": [1, 2]}
    row = bekko_row(state=state,
                    decisions=[decision("n", "noul", ("false", "true"),
                                        ("F", "T"), instructions="P.")],
                    targets=[target("n", ("false", "true"), (1.0, 0.0))])
    records, _ = map_bekko_row(row, "cfg", "r" * 40)
    assert records[0]["request"]["state"] == json.dumps(state, ensure_ascii=False)


# --- acceptance guard ---------------------------------------------------------


def noul_row(case_id, group, instructions="P.", true_prob=1.0, state=None):
    return bekko_row(case_id=case_id, group=group, state=state,
                     decisions=[decision("n", "noul", ("false", "true"),
                                         ("F", "T"), instructions=instructions)],
                     targets=[target("n", ("false", "true"),
                                     (1.0 - true_prob, true_prob))])


def accepted_records(rows, config="cfg", tokenizer=None, blocked=frozenset(),
                     max_length=8192, budget=16384):
    accept, counters = make_guard(tokenizer or FakeTokenizer(), blocked,
                                  max_length, budget)
    kept = []
    for row in rows:
        records, row_counters = map_bekko_row(row, config, "r" * 40)
        counters.update(row_counters)
        for record in records:
            if accept(record, "openjev" if config.startswith("open_jev__")
                      else "general"):
                kept.append(record)
    return kept, counters


def test_overlength_rejects_whole_question_but_keeps_sibling():
    long_options = ("detail " * 80, "detail " * 80)  # only this question is long
    row = bekko_row(
        state={"scene": "small scene"},
        decisions=[decision("big", "choice", ("a", "b"), long_options),
                   decision("n", "noul", ("false", "true"), ("F", "T"),
                            instructions="P.")],
        targets=[target("big"), target("n", ("false", "true"), (1.0, 0.0))])
    kept, counters = accepted_records([row], max_length=64)
    assert [qid for record in kept for qid in record["request"]["questions"]] == ["n"]
    assert counters["overlength_whole_question"] == 1


def test_question_token_budget_rejects_complete_question():
    longish = "filler " * 30  # ~180 tokens per leaf
    row = bekko_row(
        state={"scene": longish},
        decisions=[decision("q", "choice", ("a", "b"), (longish, longish))],
        targets=[target("q", ("a", "b"), (0.5, 0.5))])
    from jevforge.encode import encode_examples
    guard_records, _ = map_bekko_row(row, "cfg", "r" * 40)
    cost = padded_question_cost(
        encode_examples(guard_records[0]["request"], FakeTokenizer(), 8192)[0])
    assert cost > 128
    kept, counters = accepted_records([row], budget=128)
    assert kept == []
    assert counters["overlength_whole_question"] == 1


def test_exclusion_and_cross_split_and_dedup_guards():
    shared_state = {"dup": "same"}
    rows = [
        noul_row("c1", find_group("t", "train"), state=shared_state),
        # Same canonical state, group hashed to a different partition: dropped.
        noul_row("c2", find_group("d", "dev"), state=shared_state,
                 instructions="P2."),
        noul_row("c3", find_group("d2", "dev"), state={"other": "state"}),
    ]
    kept, counters = accepted_records(rows)
    assert len(kept) == 2
    assert counters["cross_split_state"] == 1
    assert {record["id"] for record in kept} == {"cfg:c1:n", "cfg:c3:n"}

    # A duplicated full input is dropped globally.
    twin = json.loads(json.dumps(rows[0]))
    twin["case_id"] = "twin"
    kept, counters = accepted_records([rows[0], twin])
    assert len(kept) == 1
    assert counters["duplicate_input"] == 1


def test_canonical_state_exclusion_blocks_regardless_of_key_order(tmp_path):
    row = noul_row("c1", find_group("t", "train"), state={"k1": "v1", "k2": 2})
    exclude = tmp_path / "exclude.jsonl"
    exclude.write_text(json.dumps({"k2": 2, "k1": "v1"}) + "\n", encoding="utf-8")
    hashes, provenance, digest = load_exclude_states(exclude)
    assert len(hashes) == 1 and len(digest) == 64
    kept, counters = accepted_records([row], blocked=hashes)
    assert kept == []
    assert counters["canonical_evaluation_state_overlap"] == 1


def test_exclude_states_accepts_raw_canonical_and_wrapper_forms(tmp_path):
    exclude = tmp_path / "exclude.jsonl"
    exclude.write_text("\n".join([
        json.dumps({"a": 1, "b": "s"}),                      # structured state
        json.dumps('{"b": "s", "a": 1}'),                    # raw string form
        json.dumps({"state": {"a": 1, "b": "s"},
                    "provenance": {"repo": "x", "revision": "y" * 40}}),
        "# comment line",
    ]) + "\n", encoding="utf-8")
    hashes, provenance, _ = load_exclude_states(exclude)
    assert len(hashes) == 1
    assert provenance == [{"repo": "x", "revision": "y" * 40}]
    assert hashes == {state_hash({"b": "s", "a": 1})}


# --- Mind2Web role mapping ----------------------------------------------------


def mind2web_record(split="train", questions=None, rid="m1", group="siteA"):
    tag = rid
    questions = questions or {
        "action": {"type": "choice",
                   "instructions": f"Which element next for {tag}?",
                   "criteria": {"e1": f"button A {tag}", "e2": f"button B {tag}"}},
        "is_target": {"type": "noul",
                      "instructions": f"e1 is the next target for {tag}.",
                      "criteria": {"true": "yes", "false": "no"}},
    }
    targets = {"action": {"e1": 1.0, "e2": 0.0},
               "is_target": {"false": 0.0, "true": 1.0}}
    return {"id": rid, "schema_version": "jevforge-record-v1", "split": split,
            "source_group": group,
            "request": {"state": json.dumps({"elements": [tag, group, split]},
                                            ensure_ascii=False),
                        "questions": questions},
            "targets": targets,
            "target_kinds": {"action": "uniform_over_positive_elements",
                             "is_target": "deterministic_truth"},
            "meta": {}}


def test_mind2web_split_roles_preserved_and_test_roles_skipped():
    records, counters = map_mind2web_record(mind2web_record("train"), "r" * 40)
    assert len(records) == 2
    assert {record["split"] for record in records} == {"train"}
    assert {record["source_group"] for record in records} == {"mind2web:siteA"}
    assert records[0]["provenance"]["original_split"] == "train"
    assert records[0]["targets"]["action"] == {"e1": 1.0, "e2": 0.0}
    assert records[1]["target_kinds"]["is_target"] == "deterministic_truth"
    records, counters = map_mind2web_record(mind2web_record("test"), "r" * 40)
    assert records == [] and counters["mind2web_role_skipped"] == 1
    records, _ = map_mind2web_record(mind2web_record("ood"), "r" * 40)
    assert records == []


# --- quota selection ------------------------------------------------------------


def make_pool(counts, split="train", source="openjev", kind="noul"):
    pool = collections.defaultdict(list)
    for config, count in counts.items():
        for index in range(count):
            record = mind2web_record(rid=f"{config}-{index}",
                                     group=f"{config}-g{index}")
            record["request"]["questions"] = {
                "n": {"type": kind, "instructions": f"{index}",
                      "criteria": {"false": "F", "true": "T"}}}
            record["targets"] = {"n": {"false": 0.0, "true": 1.0}}
            record["provenance"] = {
                "case_id": f"{config}-{index}", "config": config,
                "mixture_source": source,
                "canonical_state_hash": state_hash(record["request"]["state"]),
            }
            pool[split, source, kind, config].append(record)
    return pool


def test_select_quotas_round_robin_and_shortage_failure():
    pool = make_pool({"cfgA": 4, "cfgB": 4})
    picked, info = select_quotas(pool, "train", "openjev", 6, seed=42)
    assert len(picked) == 6 and info["oversampled_copies"] == 0
    sources = collections.Counter(
        record["provenance"]["case_id"].split("-")[0] for record in picked)
    assert set(sources) == {"cfgA", "cfgB"}
    with pytest.raises(RuntimeError, match="quota shortage"):
        select_quotas(pool, "train", "openjev", 100, seed=42)


def test_select_quotas_oversample_only_for_train():
    pool = make_pool({"cfgA": 3})
    picked, info = select_quotas(pool, "train", "openjev", 6, seed=42,
                                 oversample=2)
    assert len(picked) == 6 and info["oversampled_copies"] == 3
    # Oversampling never applies outside the train split.
    with pytest.raises(RuntimeError, match="quota shortage"):
        select_quotas(make_pool({"cfgA": 3}, split="dev"), "dev", "openjev", 6,
                      seed=42, oversample=2)


# --- parquet scanning and end-to-end build ---------------------------------------


@pytest.fixture(scope="module")
def snapshot(tmp_path_factory):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path_factory.mktemp("snapshot")
    (root / "bekko").mkdir()
    (root / "mind2web" / "data").mkdir(parents=True)

    criteria = pa.struct([("id", pa.string()),
                          ("description_json", pa.string()),
                          ("value", pa.float64())])
    decision_type = pa.struct([
        ("id", pa.string()), ("kind", pa.string()), ("type", pa.string()),
        ("instructions_json", pa.string()), ("system_prompt", pa.string()),
        ("criteria", pa.list_(criteria)),
        ("documents", pa.list_(pa.struct([("id", pa.string()),
                                          ("content_json", pa.string())]))),
        ("scoring", pa.string())])
    schema = pa.schema([
        ("input", pa.struct([("state_json", pa.string()),
                             ("decisions", pa.list_(decision_type))])),
        ("targets", pa.list_(pa.struct([
            ("decision_id", pa.string()), ("kind", pa.string()),
            ("ids", pa.list_(pa.string())),
            ("probabilities", pa.list_(pa.float64())),
            ("annotation_kind", pa.string()),
            ("metadata_json", pa.string())]))),
        ("split", pa.string()), ("case_id", pa.string()),
        ("group_id", pa.string()), ("language", pa.string()),
        ("input_hash", pa.string())])

    def write(name, rows):
        path = root / "bekko" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        return str(path.relative_to(root / "bekko"))

    train_group = find_group("bt", "train")
    dev_group = find_group("bd", "dev")
    cal_group = find_group("bc", "calibration")
    openjev_rows, general_rows = [], []
    for index in range(6):
        openjev_rows.append(bekko_row(
            case_id=f"oj{index}", group=train_group, language="und",
            decisions=[decision("maneuver", "choice",
                                ("o0", "o1", "o2"), ("L", "H", "R")),
                       decision("safe", "noul", ("false", "true"), ("F", "T"))],
            targets=[target("maneuver", ("o0", "o1", "o2"),
                            (0.2, 0.5, 0.3), "synthetic_reference_distribution"),
                    target("safe", ("false", "true"), (0.25, 0.75))],
            state={"obs": f"drone scene {index}"}, input_hash=f"ojh{index}"))
        general_rows.append(bekko_row(
            case_id=f"ge{index}", group=train_group,
            decisions=[decision("cls", "noul", ("false", "true"), ("F", "T"),
                                instructions="Classify."),
                       decision("rating", "score", ("0", "1", "2"),
                                ("lo", "mid", "hi"), values=(0, 1, 2))],
            targets=[target("cls", ("false", "true"), (0.6, 0.4)),
                     target("rating", ("0", "1", "2"), (0.2, 0.3, 0.5))],
            state={"text": f"document {index}"}, input_hash=f"geh{index}"))
    openjev_rows += [
        bekko_row(case_id="oj_dev", group=dev_group, language="und",
                  decisions=[decision("safe", "noul", ("false", "true"),
                                      ("F", "T"))],
                  targets=[target("safe", ("false", "true"), (0.1, 0.9))],
                  state={"obs": "dev scene"}),
        bekko_row(case_id="oj_dev2", group=find_group("bd2", "dev"),
                  language="und",
                  decisions=[decision("safe", "noul", ("false", "true"),
                                      ("F", "T"))],
                  targets=[target("safe", ("false", "true"), (0.3, 0.7))],
                  state={"obs": "dev scene two"}),
        bekko_row(case_id="oj_cal", group=cal_group, language="und",
                  decisions=[decision("safe", "noul", ("false", "true"),
                                      ("F", "T"))],
                  targets=[target("safe", ("false", "true"), (0.8, 0.2))],
                  state={"obs": "cal scene"}),
        bekko_row(case_id="oj_bad", group=train_group, language="und",
                  decisions=[decision("bad", "score", ("0", "1"), ("x", "y"),
                                      values=(1.0, 2.0))],
                  targets=[target("bad", ("0", "1"), (0.5, 0.5))],
                  state={"obs": "nonindex scene"}),
        bekko_row(case_id="oj_split", group=train_group, language="und",
                  split="validation",
                  decisions=[decision("safe", "noul", ("false", "true"),
                                      ("F", "T"))],
                  targets=[target("safe", ("false", "true"), (0.5, 0.5))],
                  state={"obs": "wrong role"}),
    ]
    general_rows.append(bekko_row(
        case_id="ge_bad", group=train_group,
        decisions=[decision("q", "choice", tuple(f"o{i}" for i in range(33)),
                            tuple(f"d{i}" for i in range(33)))],
        targets=[target("q", tuple(f"o{i}" for i in range(33)),
                        [1.0] + [0.0] * 32)],
        state={"text": "oversize candidates"}))

    eval_row = bekko_row(
        case_id="evalcase", group="evalgroup", split="train",
        decisions=[decision("q", "noul", ("false", "true"), ("F", "T"))],
        targets=[target("q")], state={"secret": "eval"})
    files = {
        "open_jev__tiny": [write("data/open_jev__tiny/train/000.parquet",
                                 openjev_rows)],
        "general__tiny": [write("data/general__tiny/train/000.parquet",
                                general_rows)],
        "eval_only": [write("data/eval_only/train/000.parquet", [eval_row])],
    }
    (root / "bekko" / "training-manifest.json").write_text(json.dumps({
        "train": [{"dataset": config, "split": "train", "data_files": paths}
                  for config, paths in files.items()],
        "evaluation": [{"dataset": "eval_only", "split": "test",
                        "data_files": files["eval_only"]}],
    }), encoding="utf-8")
    (root / "bekko" / "sources.json").write_text(json.dumps({
        "note": "No blanket relicensing.",
        "datasets": {"open_jev__tiny": {"train_and_validation": {
            "upstream_datasets": [{"license": ["CC0-1.0"]}]}}}}),
        encoding="utf-8")

    for split in ("train", "dev", "calibration"):
        records = [mind2web_record(split, rid=f"m2w-{split}-{index}",
                                   group=f"site-{split}")
                   for index in range(4)]
        (root / "mind2web" / "data" / f"{split}.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8")
    (root / "mind2web" / "SOURCE.json").write_text(json.dumps(
        {"dataset": "LangAGI-Lab/Mind2Web-axtree-cleaned-lite",
         "upstream_license": "CC-BY-4.0"}), encoding="utf-8")

    configs = root / "configs.json"
    configs.write_text(json.dumps(["open_jev__tiny", "general__tiny"]),
                       encoding="utf-8")
    exclude = root / "exclude.jsonl"
    exclude.write_text(json.dumps({"obs": "drone scene 0"}) + "\n",
                       encoding="utf-8")
    return {"root": root, "configs": str(configs), "exclude": str(exclude)}


def test_iter_bekko_rows_respects_scan_cap(snapshot):
    rows = list(iter_bekko_rows(snapshot["root"] / "bekko",
                                ["data/open_jev__tiny/train/000.parquet"],
                                scan_rows=3, seed=42))
    assert len(rows) == 3
    again = list(iter_bekko_rows(snapshot["root"] / "bekko",
                                 ["data/open_jev__tiny/train/000.parquet"],
                                 scan_rows=3, seed=42))
    assert rows == again


def run_build(snapshot, output, **overrides):
    parameters = dict(
        bekko_dir=snapshot["root"] / "bekko", bekko_revision="9c" * 20,
        mind2web_dir=snapshot["root"] / "mind2web", mind2web_revision="8c" * 20,
        configs=["open_jev__tiny", "general__tiny"], tokenizer=FakeTokenizer(),
        output_dir=output, train_decisions=12, dev_decisions=3,
        calibration_decisions=1, max_length=8192, max_candidates=32,
        question_token_budget=16384, seed=42,
        exclude_states=snapshot["exclude"], cap_openjev=50, cap_general=50,
        scan_rows=5000)
    parameters.update(overrides)
    return build(**parameters)


def test_build_end_to_end(snapshot, tmp_path):
    output = tmp_path / "out"
    manifest = run_build(snapshot, output)
    # Exact quotas, mixture ratios and role enforcement.
    assert manifest["quotas"]["train"] == \
        {"openjev": 7, "mind2web": 3, "general": 2}
    splits = {split: load_records(output / f"{split}.jsonl")
              for split in ("train", "dev", "calibration")}
    assert len(splits["train"]) == 12 and len(splits["dev"]) == 3
    assert len(splits["calibration"]) == 1
    sources = collections.Counter(
        record["provenance"]["mixture_source"] for record in splits["train"])
    assert sources == {"openjev": 7, "mind2web": 3, "general": 2}
    # Evaluation-role config never read; only requested configs appear.
    configs_used = {record["provenance"]["config"]
                    for records in splits.values() for record in records}
    assert "eval_only" not in configs_used
    # Split isolation: no shared group or canonical state across outputs.
    for first, second in (("train", "dev"), ("train", "calibration"),
                          ("dev", "calibration")):
        assert not ({r["source_group"] for r in splits[first]} &
                    {r["source_group"] for r in splits[second]})
        assert not ({r["provenance"]["canonical_state_hash"]
                     for r in splits[first]} &
                    {r["provenance"]["canonical_state_hash"]
                     for r in splits[second]})
    # Explicit rejection counters recorded in the manifest.
    rejections = manifest["rejections"]
    assert rejections["rejected:nonindex_score_scale"] == 1
    assert rejections["rejected:candidate_count"] == 1
    assert rejections["not_train"] == 1
    assert rejections["canonical_evaluation_state_overlap"] == 2  # drone 0: both decisions
    assert manifest["score_policy"].startswith("Bekko retains only exact")
    assert manifest["license_note"] == "No blanket relicensing."
    assert manifest["source_terms"]["open_jev__tiny"] == \
        [{"license": ["CC0-1.0"]}]
    assert manifest["mind2web_upstream"]["upstream_license"] == "CC-BY-4.0"
    assert manifest["evaluation_input_exclusion"]["labels_or_scores_used"] is False
    assert set(manifest["source_input_files_sha256"]) >= {
        str(snapshot["root"] / "mind2web" / "data" / "train.jsonl")}


def test_build_is_deterministic(snapshot, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    run_build(snapshot, first)
    run_build(snapshot, second)
    for split in ("train", "dev", "calibration"):
        assert (first / f"{split}.jsonl").read_bytes() == \
            (second / f"{split}.jsonl").read_bytes()
    assert json.loads((first / "manifest.json").read_text()) == \
        json.loads((second / "manifest.json").read_text())


def test_build_quota_shortage_fails_clearly(snapshot, tmp_path):
    with pytest.raises(RuntimeError, match="quota shortage"):
        run_build(snapshot, tmp_path / "out", dev_decisions=300)
