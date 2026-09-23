# Contributing to JevForge

Thanks for your interest! JevForge is a young project — issues and PRs are welcome.

- **Bugs / weird metrics**: open an issue with the command you ran and the relevant
  `artifacts/*.json`. Fixed split hashes live in `artifacts/fixed_splits.json`.
- **New domains** (webarena, GUI, RAG routing…): implement a `build_<domain>.py` that emits
  `jevforge-record-v1` JSONL — the rest of the pipeline (augment/train/serve/evaluate) is domain-agnostic.
- **Model experiments** (competition heads, prefix-KV inference, RLCD arm): see the v2 roadmap
  in `docs/DESIGN.md`; keep baselines comparable by evaluating on the frozen splits.
- Run `python -m jevforge.train --records <dir> --validate-only`-style smoke tests before submitting.

MIT licensed; by contributing you agree your contributions are MIT licensed too.
