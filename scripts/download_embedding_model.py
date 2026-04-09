#!/usr/bin/env python3
"""Download a Hugging Face Hub repo (model/dataset/space) into a local directory."""

from __future__ import annotations

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download a Hugging Face model (or dataset/space) snapshot to a local folder."
    )
    p.add_argument(
        "--repo-id",
        "-r",
        dest="repo_id_flag",
        default="Qwen/Qwen3-Embedding-8B",
        help="Same as positional REPO_ID; avoids shells splitting org/name into two tokens.",
    )
    p.add_argument(
        "--local-dir",
        "-o",
        default="hf_home/Qwen3-Embedding-8B",
        help="Directory to save files into (created if missing). Default: %(default)s",
    )
    p.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Repository type on the Hub (default: model).",
    )
    p.add_argument(
        "--token",
        default=None,
        help="Access token (private/gated repos). If omitted, uses HF_TOKEN env when set.",
    )
    p.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Override Hub API URL, e.g. https://hf-mirror.com for mirrors. "
        "If omitted, uses HF_ENDPOINT env when set.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=20,
        help="Parallel download workers (default: %(default)s).",
    )
    p.add_argument(
        "--revision",
        default=None,
        help="Optional Git revision (branch / tag / commit) on the Hub.",
    )
    ns = p.parse_args()
    repo_id = ns.repo_id_flag or ns.repo_id
    if not repo_id:
        p.error("provide REPO_ID as a positional argument or use --repo-id / -r")
    ns.repo_id = repo_id
    del ns.repo_id_flag
    return ns


def main() -> int:
    args = parse_args()
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint.rstrip("/")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub. Install with: pip install huggingface_hub", file=sys.stderr)
        return 1

    token = args.token or os.environ.get("HF_TOKEN")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=args.local_dir,
        token=token,
        max_workers=args.max_workers,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
