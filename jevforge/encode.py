"""JevForge prompt encoding (own markup, shared by training and inference).

Each candidate path is

    <|oj:state|>
    {state}
    <|oj:question|>
    {type}: {instructions}
    [yes := ... / no := ...]          (noul only, when criteria present)
    {candidate rendering}
    <|oj:answer|>

and the decision readout is the hidden state at the final <|oj:answer|>
marker. Choice candidates render as "[key] description", score levels as
"level i: description", noul paths are the literal "no"/"yes" words.
"""
from .schema import candidate_ids

STATE_MARK = "<|oj:state|>"
QUESTION_MARK = "<|oj:question|>"
ANSWER_MARK = "<|oj:answer|>"


def render_candidate(question, index):
    kind = question["type"]
    if kind == "choice":
        key = list(question["criteria"])[index]
        return f"[{key}] {question['criteria'][key]}"
    if kind == "score":
        return f"level {index}: {question['criteria'][index]}"
    return "yes" if index == 1 else "no"


def question_prefix(state, question):
    lines = [STATE_MARK, state, QUESTION_MARK,
             f"{question['type']}: {question['instructions']}"]
    criteria = question.get("criteria")
    if question["type"] == "noul" and isinstance(criteria, dict):
        if "true" in criteria:
            lines.append(f"yes := {criteria['true']}")
        if "false" in criteria:
            lines.append(f"no := {criteria['false']}")
    return "\n".join(lines) + "\n"


def encode_leaves(state, question, tokenizer):
    prefix = tokenizer.encode(question_prefix(state, question), add_special_tokens=False)
    leaves = []
    for index in range(len(candidate_ids(question))):
        segment = render_candidate(question, index) + "\n" + ANSWER_MARK
        leaves.append(prefix + tokenizer.encode(segment, add_special_tokens=False))
    return leaves


def encode_examples(request, tokenizer, max_length):
    examples = []
    for qid, question in request["questions"].items():
        leaves = encode_leaves(request["state"], question, tokenizer)
        longest = max(len(leaf) for leaf in leaves)
        if longest > max_length:
            raise ValueError(
                f"question {qid}: longest candidate path {longest} exceeds "
                f"max_length={max_length}; refusing to truncate")
        examples.append({"question_id": qid, "type": question["type"],
                         "candidate_ids": candidate_ids(question), "leaves": leaves})
    return examples
