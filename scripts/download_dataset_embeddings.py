#!/usr/bin/env python3
"""
从 Hugging Face Hub 下载公开数据集及其embedding（snapshot_download）。
aceback 时设置: HF_DOWNLOAD_VERBOSE=1
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ID = "zxnn060606/TESS"

LOCAL_DIR = Path(__file__).resolve().parent.parent / "dataset" 

REVISION = None

HF_MIRROR_ENDPOINT = "https://hf-mirror.com"

_WORKER_ARG = "--hf-download-worker"

# huggingface_hub：远端失败且 local_dir 已存在时可能仍返回 0，并打印如下含义的日志
_UNREACHABLE_BUT_RETURNED_LOCAL = "remote repo cannot be accessed"


def _worker_download() -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "未安装 huggingface-hub。\n"
            "  python -m pip install huggingface-hub",
            file=sys.stderr,
        )
        return 1

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    out = str(LOCAL_DIR.resolve())

    print(f"repo_id={REPO_ID!r}")
    print(f"local_dir={out}")
    endpoint = os.environ.get("HF_ENDPOINT", "(https://huggingface.co)")
    print(f"HF_ENDPOINT={endpoint!r}")

    try:
        path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=out,
            revision=REVISION,
            token=None,
        )
    except Exception as e:
        err_type = type(e).__name__
        first_line = (str(e) or "").strip().split("\n")[0][:400]
        print(f"snapshot_download 失败: {err_type}: {first_line}", file=sys.stderr)
        if os.environ.get("HF_DOWNLOAD_VERBOSE", "").strip():
            import traceback

            traceback.print_exc()
        return 1

    print(f"Done: {path}")
    return 0


def _subprocess_really_succeeded(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode != 0:
        return False
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if _UNREACHABLE_BUT_RETURNED_LOCAL in combined:
        return False
    return True


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == _WORKER_ARG:
        return _worker_download()

    script_path = str(Path(__file__).resolve())
    attempts: list[tuple[str, str | None]] = [
        ("Hugging Face 官方", None),
        ("hf-mirror 镜像", HF_MIRROR_ENDPOINT),
    ]

    last_rc = 1
    for name, endpoint_url in attempts:
        print(f"\n>>> 尝试通过【{name}】下载 ...")
        env = os.environ.copy()

        env.pop("HF_ENDPOINT", None)
        env.pop("HF_HUB_ENDPOINT", None)
        if endpoint_url:
            env["HF_ENDPOINT"] = endpoint_url

        proc = subprocess.run(
            [sys.executable, script_path, _WORKER_ARG],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.stdout:
            sys.stdout.write(proc.stdout)
        if proc.stderr:
            sys.stderr.write(proc.stderr)

        if _subprocess_really_succeeded(proc):
            print(f"\n已通过【{name}】下载成功。")
            return 0

        if proc.returncode == 0:
            print(
                f"\n【{name}】失败 (exit {proc.returncode})，将尝试下一端点。",
                file=sys.stderr,
            )
        else:
            print(f"\n【{name}】失败 (exit {proc.returncode})，将尝试下一端点。")
        last_rc = proc.returncode if proc.returncode != 0 else 1

    print("\n官方与镜像均失败，请检查网络、代理、repo_id 或稍后重试。", file=sys.stderr)
    return last_rc


if __name__ == "__main__":
    raise SystemExit(main())
