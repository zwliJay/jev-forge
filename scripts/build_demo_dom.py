#!/usr/bin/env python3
"""Build web/demo_dom.json: demo samples with structured, DOM-renderable elements.

Rejoins web/demo_data.json (record_id, arms, gold) with the dataset records,
parses each candidate's rendered text ("li id=x role=tab aria_label=...") back
into {tag, attrs} and classifies it as an interactive control vs decoration.
The demo page renders these as real <button>/<a>/<input>/<table> elements.
"""
import json
import re
from pathlib import Path

ATTR_KEYS = ("id", "role", "aria_label", "name", "text_value", "value",
             "placeholder", "title", "type", "alt", "label", "input_value")
OPENERS = tuple(f"{key}=" for key in ATTR_KEYS)
INTERACTIVE_TAGS = {"button", "a", "input", "select", "textarea", "option", "form"}
INTERACTIVE_ROLES = {"button", "link", "tab", "tablist", "checkbox", "radio",
                     "menuitem", "option", "searchbox", "textbox", "switch"}
LABEL_KEYS = ("aria_label", "name", "text_value", "value", "placeholder",
              "title", "alt", "label", "input_value")


def parse_element(text):
    """'li id=bookCarTab role=tab aria_label=heading level 3 ...' -> dict."""
    tokens = text.split()
    tag = (tokens[0] if tokens else "div").lower()
    attrs, current, buf = {}, None, []

    def flush():
        if current is not None:
            attrs[current] = " ".join(buf).strip()

    for token in tokens[1:]:
        opener = next((f"{k}=" for k in ATTR_KEYS if token.startswith(f"{k}=")), None)
        if opener:
            flush()
            current = opener[:-1]
            buf = [token[len(opener):]]
        elif token in ATTR_KEYS:  # bare key with empty glued value
            flush()
            current = token
            buf = []
        else:
            buf.append(token)
    flush()
    attrs.pop("", None)
    label = next((attrs[k] for k in LABEL_KEYS if attrs.get(k)), "")
    kind = "control"
    if tag in INTERACTIVE_TAGS or attrs.get("role", "") in INTERACTIVE_ROLES:
        kind = "control"
    elif attrs.get("id") or attrs.get("aria_label") or label:
        kind = "labelled"
    else:
        kind = "decor"
    return {"tag": tag, "attrs": attrs, "label": label[:90], "kind": kind}


def main():
    demo = json.loads(Path("web/demo_data.json").read_text(encoding="utf-8"))
    records = {}
    for split in ("test", "ood"):
        path = Path(f"data/web_pilot/{split}.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            records[record["id"]] = record

    out = {"meta": dict(demo["meta"]), "samples": {}}
    for split, samples in demo["samples"].items():
        built = []
        for sample in samples:
            record = records.get(sample["record_id"], {})
            state = json.loads(record.get("request", {}).get("state", "{}"))
            elements = []
            for cand in sample["candidates"]:
                parsed = parse_element(cand["text"])
                parsed["id"] = cand["id"]
                elements.append(parsed)
            built.append({
                "record_id": sample["record_id"],
                "website": sample["website"], "split": sample["split"],
                "task": sample["task"],
                "operation": state.get("operation", ""),
                "human_action": record.get("meta", {}).get("target_action_reprs", ""),
                "elements": elements,
                "gold": sample["gold"],
                "probabilities": sample["probabilities"],
            })
        out["samples"][split] = built

    Path("web/demo_dom.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    counts = {}
    for split, samples in out["samples"].items():
        controls = sum(1 for s in samples for e in s["elements"] if e["kind"] == "control")
        total = sum(len(s["elements"]) for s in samples)
        counts[split] = {"samples": len(samples), "elements": total, "controls": controls}
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
