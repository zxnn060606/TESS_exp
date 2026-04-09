# MMTSF / News Forecasting

本仓库提供时间序列与新闻等多模态实验代码。以下说明如何**选择单/多模态**、**原语（primitive）嵌入**开关、**训练入口**，以及**数据集\嵌入下载与本地生成**。

## 环境

推荐顺序：**先用 Conda 根据 `mmtsf.yml` 创建基础环境（含 Python 等），再在该环境中用 pip 安装 `requirements_mmtsf.txt`**。以下命令均在**仓库根目录**执行（含 `mmtsf.yml` 与 `requirements_mmtsf.txt` 的目录）。

```bash

conda env create -f mmtsf.yml
conda activate mmtsf
python -m pip install -r requirements_mmtsf.txt

```

---

## 单模态 vs 多模态

是否在pipeline里**启用多模态**（时间序列 + 文本），由数据集 YAML 中的 **`use_multimodal`** 控制：

| `use_multimodal`（在 `src/model_trainer/configs/dataset/<Dataset>.yaml`） | 含义 |
|------------------------------------------------------------------------|------|
| `false` | **不启用多模态**：仅按时间序列主任务组织数据与训练。 |
| `true`  | **启用多模态**：在模型中融合新闻等侧信息，通常需要预计算的新闻向量等。 |

启用多模态时，通常还需在同一数据集 yaml 里打开 **`use_news_embedding: true`**，并保证能加载到对应 `.pt` 等嵌入文件（见下文「新闻 embedding」）。

---

## 「原语」`use_primitive` 怎么开

在 ** FNSPID多模态情况下 **`use_primitive` 决定**选用什么样的文本embedding**向量喂给模型：

- **`false`**：使用 **original_text** （与 registry 中 `ver_camf` 别名对应）。
- **`true`**：使用 **primitive** 。

可在 **`src/model_trainer/configs/dataset/FNSPID.yaml`** 中设置 `use_primitive`，也可用命令行启动时直接指定

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

```bash
python scripts/download_dataset_embeddings.py
```

若官方与镜像均失败：检查网络/代理、防火墙，或在本机可访问的环境下重试。

---

## FNSPID 新闻 embedding：无法从上述脚本获得时

FNSPID 的新闻向量可由 **`scripts/generate_qwen_embeddings.py`** 在本地用Embedding模型生成

### 1）先下载嵌入模型（Hub 失败时）

脚本：`scripts/download_embedding_model.py`

默认拉取 **`Qwen/Qwen3-Embedding-8B`** 到 **`hf_home/Qwen3-Embedding-8B`**，默认使用镜像 **`https://hf-mirror.com`**

```bash

# 默认镜像 + 默认路径
python scripts/download_embedding_model.py --token your_token

```

请使用自己的hf_token填入 `--token` 或环境变量 **`HF_TOKEN`**。

### 2）本地生成 embedding

**`scripts/generate_all_embeddings.sh`**

```bash

bash scripts/generate_all_embeddings.sh
```

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


