"""Publish a completed JevForge checkpoint without exposing credentials."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def load_env_token(path: Path) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("HF_TOKEN="):
            token = raw.split("=", 1)[1].strip().strip('"').strip("'")
            if token:
                return token
    raise SystemExit(f"HF_TOKEN is missing from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or load_env_token(args.env_file)
    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(args.checkpoint),
        commit_message="Release JevForge-0.6B checkpoint and training evidence",
        ignore_patterns=["*.tmp", "*.log"],
    )
    print(result)


if __name__ == "__main__":
    main()
