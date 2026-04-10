## 环境

在**仓库根目录**执行：

```bash
conda env create -f mmtsf.yml
conda activate mmtsf
python -m pip install -r requirements_mmtsf.txt
```

---

## HuggingFace 下载

**数据集快照**（脚本内固定 Hub 数据集，保存到 `dataset/`）：

```bash
python scripts/download_dataset_embeddings.py
```

---

## 运行命令


```bash
python -m src/model_trainer.main -d FNSPID -g 0
```

参数：`--dataset` / `-d`、`--gpu` / `-g`、`--use-primitive true|false`

快速启动

```bash
bash scripts/run_train_FNSPID_multimodal_default.sh
```

**本地生成新闻向量**（模型已下载到本地后，`PYTHONPATH` 含 `src`）：

```bash
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/generate_qwen_embeddings.py --alias FNSPID/ver_primitive --model-path hf_home/Qwen3-Embedding-8B --batch-size 32 --device cuda:0
bash scripts/generate_all_embeddings.sh
```

单/多模态：改 `src/model_trainer/configs/dataset/<数据集>.yaml` 中的 `use_multimodal`、`use_news_embedding`。
