# MMTSF / News Forecasting

本仓库提供时间序列与新闻等多模态实验代码。以下说明如何**选择单/多模态**、**原语（primitive）双路嵌入**开关、**训练入口**，以及**数据集 / 嵌入模型下载与本地生成 embedding**。

## 环境

- 推荐使用仓库内 `mmtsf.yml`（Conda）或 `requirements_mmtsf.txt`（pip）安装依赖。
- 训练与脚本需在仓库根目录执行（或自行设置 `PYTHONPATH` 指向 `src`），例如：

```bash
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
```

Windows PowerShell 示例：

```powershell
$env:PYTHONPATH = "$PWD\src"
```

---

## 单模态 vs 多模态

默认**用哪个模型**由数据集 YAML 里的 `use_multimodal` 决定（见 `infer_default_model_from_dataset`）：

| `use_multimodal`（在 `src/model_trainer/configs/dataset/<Dataset>.yaml`） | 默认加载的模型配置 |
|------------------------------------------------------------------------|-------------------|
| `false` | `UniModal_Baseline` → `configs/model/UniModal_Baseline.yaml`（关闭新闻嵌入需求，DataLoader 以 `(x, y)` 为主） |
| `true`  | `MultiModal_Baseline` → `configs/model/MultiModal_Baseline.yaml`（需要新闻向量等） |

多模态训练时，通常还需在同一数据集 yaml 中打开 **`use_news_embedding: true`**，并保证数据侧能加载到对应 `.pt` 等嵌入文件（见下文「新闻 embedding」）。

> 说明：当前 `main.py` 未暴露 `-m` 选择任意模型名；若需其他模型，可在调用 `quick_start` 时传入 `model=...`，或扩展 CLI。

---

## 「原语」`use_primitive` 是什么、怎么开

在 **FNSPID 且多模态 + 双路新闻嵌入**（batch 里同时有 `news_embed_original` 与 `news_embed_ver_primitive`）时，`use_primitive` 决定**选用哪一路**向量喂给模型：

- **`false`（默认常见含义）**：使用 **original** 一路（与 registry 中 `ver_camf` 等别名对应的那套嵌入）。
- **`true`**：使用 **ver_primitive** 一路。

可在 **`src/model_trainer/configs/dataset/FNSPID.yaml`** 中设置 `use_primitive`，也可用命令行**覆盖** yaml：

```bash
python -m model_trainer.main -d FNSPID --use-primitive true -g 0
python -m model_trainer.main -d FNSPID --use-primitive false -g 0
```

不写 `--use-primitive` 时，以 yaml 为准。

---

## 训练快速启动

示例脚本（Bash，**在仓库根目录**执行）：

| 脚本 | 作用 |
|------|------|
| `scripts/run_train_FNSPID_multimodal_default.sh` | 示例：设置 `PYTHONPATH` 后启动 `python -m model_trainer.main -d FNSPID ...`；可通过 `GPU_ID=1 bash scripts/run_train_FNSPID_multimodal_default.sh` 指定 GPU；其余参数会原样传给 `main.py`。 |

等价的手动命令示例：

```bash
cd /path/to/repo
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python -m model_trainer.main -d FNSPID -g 0
```

`main.py` 支持：`--dataset` / `-d`、`--gpu` / `-g`、`--use-primitive true|false`。

---

## 下载「数据集」（含 Hub 上的数据快照）

脚本：`scripts/download_dataset_embeddings.py`

- 使用 `huggingface_hub.snapshot_download` 从 Hub 拉取**固定的一个数据集仓库**（脚本内 `REPO_ID = "zxnn060606/TESS"`），保存到 **`dataset/TESS/`**。
- 会先尝试官方端点，再尝试 **`https://hf-mirror.com`**（通过子进程设置 `HF_ENDPOINT`）。
- 需要安装：`pip install huggingface-hub`
- 详细日志可设置：`HF_DOWNLOAD_VERBOSE=1`

```bash
python scripts/download_dataset_embeddings.py
```

若官方与镜像均失败：检查网络/代理、防火墙，或在本机可访问的环境下重试；该脚本与 **FNSPID 新闻向量**无直接对应关系（FNSPID 见下一节）。

---

## FNSPID 新闻 embedding：无法从上述脚本获得时

FNSPID 的新闻向量一般由 **`scripts/generate_qwen_embeddings.py`** 在本地用句子嵌入模型生成，或由你自行放置到 registry（`src/model_trainer/configs/dataset/index.yaml`）所描述的路径下。

### 1）先下载嵌入模型（Hub 失败时）

脚本：`scripts/download_embedding_model.py`

默认拉取 **`Qwen/Qwen3-Embedding-8B`** 到 **`hf_home/Qwen3-Embedding-8B`**，默认使用镜像 **`https://hf-mirror.com`**（可用 `--endpoint` 或环境变量 `HF_ENDPOINT` 调整）。

```bash
pip install huggingface_hub

# 默认镜像 + 默认路径
python scripts/download_embedding_model.py

# 指定本地目录与仓库（示例）
python scripts/download_embedding_model.py -r Qwen/Qwen3-Embedding-8B -o hf_home/Qwen3-Embedding-8B
```

私有或门禁模型可设置 `--token` 或环境变量 **`HF_TOKEN`**。

### 2）本地生成 embedding

批量 shell（编辑其中的 `MODEL_PATH`、`DEVICE`、`DATASETS` 等）：**`scripts/generate_all_embeddings.sh`**

```bash
# 在仓库根目录
bash scripts/generate_all_embeddings.sh
```

其内部调用 `python scripts/generate_qwen_embeddings.py`，常用参数包括：

- `--alias`：数据集别名，如 `FNSPID/ver_camf`、`FNSPID/ver_primitive`（与 `index.yaml` 中别名一致）
- `--model-path`：本地嵌入模型目录（相对路径相对于**当前工作目录**，建议在仓库根执行）
- `--batch-size`、`--device`（或 `--devices`）

单条命令示例：

```bash
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/generate_qwen_embeddings.py \
  --alias FNSPID/ver_primitive \
  --model-path hf_home/Qwen3-Embedding-8B \
  --batch-size 32 \
  --device cuda:0
```

生成日志默认写在 `embedding_generation.log` 与 `logs/` 下（见 `generate_all_embeddings.sh`）。

---

## 目录速查

| 路径 | 说明 |
|------|------|
| `src/model_trainer/main.py` | 训练入口 |
| `src/model_trainer/configs/dataset/FNSPID.yaml` | FNSPID 数据与 `use_multimodal` / `use_news_embedding` / `use_primitive` 等 |
| `src/model_trainer/configs/dataset/index.yaml` | 数据集别名、划分路径、嵌入 `.pt` 相对路径 |
| `scripts/run_train_FNSPID_multimodal_default.sh` | 训练示例脚本 |
| `scripts/download_dataset_embeddings.py` | Hub 下载 TESS 数据集快照 |
| `scripts/download_embedding_model.py` | Hub 下载嵌入模型（如 Qwen3-Embedding-8B） |
| `scripts/generate_qwen_embeddings.py` | 按别名生成/写入新闻 embedding |
| `scripts/generate_all_embeddings.sh` | 对多个别名批量调用生成脚本 |

如有新的数据集别名或嵌入路径，请同步维护 **`index.yaml`** 与 **`FNSPID.yaml`** 中的可选覆盖项。
