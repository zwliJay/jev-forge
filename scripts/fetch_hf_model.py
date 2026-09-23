"""Download a Hugging Face model snapshot using an env-file token if present."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def token_from_env_file(path: Path) -> str | None:
    if not path.exists():
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("HF_TOKEN="):
            return raw.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        cache_dir=str(args.cache_dir),
        token=os.environ.get("HF_TOKEN") or token_from_env_file(args.env_file),
    )
    print(path)


if __name__ == "__main__":
    main()
