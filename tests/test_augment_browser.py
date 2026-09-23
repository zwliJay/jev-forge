import json

from jevforge.augment_browser import augment_record
from jevforge.schema import validate_record


def example_record():
    elements = [
        {"id": "gold", "text": "button text=Checkout"},
        {"id": "hard", "text": "button text=Checkout later"},
        {"id": "easy", "text": "link text=Help"},
    ]
    return {
        "id": "shop-step-1", "schema_version": "jevforge-record-v1",
        "split": "train", "source_group": "shop",
        "request": {
            "state": json.dumps({"task": "Checkout the cart", "elements": elements}),
            "questions": {"action": {
                "type": "choice", "instructions": "Pick the next element",
                "criteria": {item["id"]: item["text"] for item in elements},
            }},
        },
        "targets": {"action": {"gold": 1.0, "hard": 0.0, "easy": 0.0}},
        "target_kinds": {"action": "uniform_over_positive_elements"},
        "gold": {"action": "gold"}, "meta": {},
    }


def test_augmentation_keeps_gold_and_prefers_hard_negative():
    views = augment_record(example_record(), views=1, negatives_per_view=1, seed=41)
    assert len(views) == 1
    augmented = validate_record(views[0])
    assert list(augmented["targets"]["action"]) == ["gold", "hard"]
    assert augmented["targets"]["action"]["gold"] == 1.0
    state = json.loads(augmented["request"]["state"])
    assert [element["id"] for element in state["elements"]] == ["gold", "hard"]
