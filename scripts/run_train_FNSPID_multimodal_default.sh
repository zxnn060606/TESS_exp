#!/usr/bin/env bash
# 启动训练：与 configs/dataset/FNSPID.yaml 中默认多模态配置一致（约 14–17 行）：
#   use_primitive: false      → 双嵌入时选用 original（ver_camf），不用 ver_primitive
#   use_multimodal: true
#   use_text_news: true
#   use_news_embedding: true
#
# 用法（在仓库根目录，或任意路径执行本脚本）：
#   bash scripts/run_train_FNSPID_multimodal_default.sh
#   GPU_ID=1 bash scripts/run_train_FNSPID_multimodal_default.sh
# 其余参数会原样传给 main.py，例如：bash scripts/run_train_FNSPID_multimodal_default.sh --help
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m model_trainer.main -d FNSPID --use-primitive true -g "${GPU_ID:-0}" "$@"
