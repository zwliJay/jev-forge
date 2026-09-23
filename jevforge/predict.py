"""JevForge predictor: checkpoint -> Jev-contract answers.

Loads the trained decision model once and answers public-contract requests
({state, questions}) with complete probability distributions. Candidate paths
share the encoded prefix through the backbone KV cache during inference, so
one question costs one prefix forward plus K short suffix forwards.
"""
import argparse
import json
import math
from pathlib import Path

from .encode import encode_examples
from .schema import candidate_ids, load_records, validate_question


def confidence_from(probabilities):
    """Shape-derived confidence: 0 for a uniform distribution, 1 for one-hot."""
    k = len(probabilities)
    if k < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probabilities) - 1.0 / k) * k / (k - 1)))


class Predictor:
    def __init__(self, checkpoint_dir, device_name="auto", precision="auto"):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        root = Path(checkpoint_dir).resolve(strict=True)
        run_config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        backbone_dir = root / "backbone"
        if device_name == "auto":
            if torch.cuda.is_available():
                device_name = "cuda:0"
            elif torch.backends.mps.is_available():
                device_name = "mps"
            else:
                device_name = "cpu"
        self.device = torch.device(device_name)
        if precision == "auto":
            precision = "bf16" if self.device.type == "cuda" else (
                "fp16" if self.device.type == "mps" else "fp32")
        self.precision = precision
        self.model_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[precision]
        raw_temperature = run_config.get("temperature", 1.0)
        if isinstance(raw_temperature, dict):  # per-type calibration
            self.temperature = {k: float(v) for k, v in raw_temperature.items()}
        else:
            self.temperature = float(raw_temperature)
        self.max_length = int(run_config.get("max_length", 512))
        self.base_name = Path(run_config["base_model"]).name

        backbone_config = AutoConfig.from_pretrained(str(backbone_dir),
                                                     local_files_only=True,
                                                     trust_remote_code=False)
        backbone_config.use_cache = True
        backbone = AutoModel.from_config(backbone_config,
                                         attn_implementation="sdpa",
                                         trust_remote_code=False)
        self.model = type("Shim", (), {})()  # placeholder for typing clarity
        from .model import JevForgeModel, backbone_hidden_size

        self.model = JevForgeModel(backbone, backbone_hidden_size(backbone_config))
        weights = load_file(str(root / "best.safetensors"), device="cpu")
        weights = {k.replace("._orig_mod.", "."): v for k, v in weights.items()}
        self.model.load_state_dict(weights, strict=True)
        self.model.to(device=self.device, dtype=self.model_dtype)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(backbone_dir),
                                                       local_files_only=True,
                                                       trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._torch = torch
        self.calls = 0

    def _forward(self, examples):
        """Batched leaf forward (right padding); returns logits per leaf."""
        torch = self._torch
        leaves, owner = [], []
        for qi, example in enumerate(examples):
            for leaf in example["leaves"]:
                leaves.append(leaf)
                owner.append(qi)
        width = max(len(leaf) for leaf in leaves)
        input_ids = torch.full((len(leaves), width), self.tokenizer.pad_token_id,
                               dtype=torch.long)
        attention = torch.zeros((len(leaves), width), dtype=torch.long)
        for i, leaf in enumerate(leaves):
            input_ids[i, :len(leaf)] = torch.tensor(leaf, dtype=torch.long)
            attention[i, :len(leaf)] = 1
        input_ids, attention = input_ids.to(self.device), attention.to(self.device)
        autocast_enabled = self.device.type == "cuda" and self.precision == "bf16"
        with torch.inference_mode(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            logits = self.model(input_ids, attention).float()
        owner = torch.tensor(owner, device=self.device)
        return [logits[owner == qi] for qi in range(len(examples))]

    def temperature_for(self, kind):
        if isinstance(self.temperature, dict):
            return self.temperature.get(kind, 1.0)
        return self.temperature

    def decide(self, state, questions, temperature=None):
        """Public Jev contract: state + questions -> per-question answers."""
        override = None if temperature is None else float(temperature)
        prepared = {}
        for qid, question in questions.items():
            validate_question(qid, question)
            prepared[qid] = dict(question)
        payload = {"state": state, "questions": prepared}
        examples = encode_examples(payload, self.tokenizer, self.max_length)
        self.calls += 1
        grouped = self._forward(examples)
        answers = {}
        for example, logits in zip(examples, grouped):
            heat = override if override is not None else self.temperature_for(example["type"])
            probabilities = (logits / heat).softmax(dim=-1).cpu().tolist()
            values = [p if math.isfinite(p) else 0.0 for p in probabilities]
            total = math.fsum(values) or 1.0
            values = [v / total for v in values]
            ids = example["candidate_ids"]
            if example["type"] == "noul":
                answers[example["question_id"]] = {"type": "noul", "noul": values[1]}
            elif example["type"] == "choice":
                best = max(range(len(ids)), key=values.__getitem__)
                answers[example["question_id"]] = {
                    "type": "choice", "choice": ids[best],
                    "probabilities": dict(zip(ids, values)),
                    "confidence": confidence_from(values),
                }
            else:
                score = math.fsum(i * v for i, v in enumerate(values))
                answers[example["question_id"]] = {
                    "type": "score", "score": score,
                    "legend": {str(i): text for i, text in enumerate(
                        payload["questions"][example["question_id"]]["criteria"])},
                    "probabilities": {str(i): v for i, v in enumerate(values)},
                    "confidence": confidence_from(values),
                }
        return answers

    def predict_records(self, records):
        """Eval helper: one row per question with probabilities and targets."""
        rows = []
        for record in records:
            examples = encode_examples(record["request"], self.tokenizer,
                                       self.max_length)
            grouped = self._forward(examples)
            for example, logits in zip(examples, grouped):
                temperature = self.temperature_for(example["type"])
                probabilities = (logits / temperature).softmax(dim=-1).cpu().tolist()
                rows.append({
                    "record_id": record["id"],
                    "question_id": example["question_id"],
                    "type": example["type"],
                    "candidate_ids": example["candidate_ids"],
                    "probabilities": probabilities,
                    "target": record["targets"][example["question_id"]],
                    "source_group": record["source_group"],
                    "gold": (record.get("gold") or {}).get(example["question_id"]),
                })
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--splits", default="dev,calibration,test,ood")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto",
                        help="auto, cpu, mps, or a CUDA device such as cuda:0")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"),
                        default="auto")
    args = parser.parse_args()

    path = Path(args.records)
    files = [path] if path.is_file() else [path / f"{s}.jsonl" for s in
                                           args.splits.split(",")]
    predictor = Predictor(args.checkpoint_dir, device_name=args.device,
                          precision=args.precision)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, file in zip(args.splits.split(","), files):
        if not file.exists():
            continue
        rows = predictor.predict_records(load_records(file))
        destination = out / f"predictions_{split}.jsonl"
        destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                       for r in rows), encoding="utf-8")
        summary[split] = len(rows)
        print(json.dumps({"split": split, "questions": len(rows),
                          "output": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
