import json
from typing import Optional

import numpy as np
import torch
from logging import getLogger
import os
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from model_trainer.utils.lag_sampling import config_lookup


def _load_split_news_tensor(
    dataset_path: str,
    embed_rel_path: Optional[str],
    flag: str,
    config,
    n_samples: int,
    *,
    config_key_for_tensor_name: Optional[str] = None,
    embed_key_config_suffix: Optional[str] = None,
) -> torch.Tensor:
    """从 path::tensor_key 或 .npy/.pt 字典加载与划分等长的新闻嵌入张量。"""
    if not embed_rel_path:
        raise ValueError(f"配置缺少 {config_key_for_tensor_name or 'news_embed'}，无法加载新闻嵌入")
    embed_rel_actual = embed_rel_path
    embed_tensor_key = None
    if "::" in embed_rel_path:
        embed_rel_actual, embed_tensor_key = embed_rel_path.split("::", 1)
        embed_tensor_key = embed_tensor_key.strip() or None
    rel_embed_path = embed_rel_actual.lstrip("/")
    embed_path = os.path.abspath(os.path.join(dataset_path, rel_embed_path))
    if not os.path.isfile(embed_path):
        raise FileNotFoundError(f"缺少新闻嵌入文件: {embed_path}")

    if embed_path.endswith((".pt", ".pth")):
        embed_dict = torch.load(embed_path, map_location="cpu")
        if not isinstance(embed_dict, dict):
            raise TypeError(f"新闻嵌入文件应为 dict 键→张量: {embed_path}")
        if embed_tensor_key is None:
            default_keys = {"train": "train_news", "vali": "vali_news", "test": "test_news"}
            key_candidates = []
            if embed_key_config_suffix:
                key_candidates.append(f"{flag}_news_embed_key_{embed_key_config_suffix}")
            key_candidates.append(f"{flag}_news_embed_key")
            embed_tensor_key = None
            for cfg_key in key_candidates:
                embed_tensor_key = _config_lookup(config, cfg_key, None)
                if embed_tensor_key:
                    break
            if not embed_tensor_key:
                embed_tensor_key = default_keys.get(flag)
        if not embed_tensor_key:
            raise KeyError(
                f"无法解析 {embed_path} 中的张量键：请在路径中使用 path::tensor_key，"
                f"或配置 {flag}_news_embed_key"
                + (f" / {flag}_news_embed_key_{embed_key_config_suffix}" if embed_key_config_suffix else "")
            )
        if embed_tensor_key not in embed_dict:
            avail = list(embed_dict.keys())
            show = avail[:24]
            more = f" …(+{len(avail) - 24})" if len(avail) > 24 else ""
            raise KeyError(
                f"在 {embed_path} 中未找到键 {embed_tensor_key!r}；文件中的键: {show}{more}"
            )
        raw_embeds = embed_dict[embed_tensor_key]
        if isinstance(raw_embeds, dict) and "embeddings" in raw_embeds:
            raw_embeds = raw_embeds["embeddings"]
        if isinstance(raw_embeds, np.ndarray):
            raw_embeds = torch.from_numpy(raw_embeds)
        raw_embeds = raw_embeds.detach().clone().cpu().type(torch.FloatTensor)
    else:
        raw_np = np.load(embed_path, allow_pickle=True)
        raw_embeds = torch.from_numpy(raw_np).type(torch.FloatTensor)

    if len(raw_embeds) != n_samples:
        raise ValueError(
            f"新闻嵌入数量与样本数不一致: len(embeds)={len(raw_embeds)}, n_samples={n_samples}"
        )
    return raw_embeds


def _config_lookup(config_obj, key: str, default=None):
    """兼容旧版调用的配置查询包装器。"""
    return config_lookup(config_obj, key, default)



class FnspidDataset(Dataset):
    def __init__(self, config, flag: str = "train", scaler=None):
        self.config = config
        self.logger = getLogger()
        base_multimodal = bool(_config_lookup(config, "use_multimodal", False))
        self.use_news_embedding = bool(_config_lookup(config, "use_news_embedding", False))
        self.use_multimodal = base_multimodal
        self.price_mode = str(_config_lookup(config, "price_mode", "normal")).lower()
        self.news_mode = str(_config_lookup(config, "news_mode", "normal")).lower()

        data_path = _config_lookup(config, "data_path", os.getcwd())
        dataset_name = _config_lookup(config, "dataset", "")
        dataset_root = _config_lookup(config, "dataset_root", None)
        if dataset_root:
            dataset_path = os.path.abspath(dataset_root)
        else:
            dataset_path = (
                os.path.abspath(os.path.join(data_path, dataset_name))
                if dataset_name
                else os.path.abspath(data_path)
            )

        def _resolve_relative(rel_path: Optional[str]) -> str:
            if rel_path is None:
                raise ValueError("Missing dataset split path")
            rel = rel_path.lstrip("/")
            return os.path.join(dataset_path, rel)

        if flag == "train":
            self.file_path = _resolve_relative(_config_lookup(config, "train_file"))
        elif flag == "vali":
            self.file_path = _resolve_relative(_config_lookup(config, "vali_file"))
        elif flag == "test":
            self.file_path = _resolve_relative(_config_lookup(config, "test_file"))
        else:
            raise ValueError(f"Unknown flag: {flag}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if scaler is None:
            self.scaler = StandardScaler()
            all_data = []
            for sample in self.data:
                hist = list(map(float, sample["historical_data"].split(",")))
                all_data.extend(hist)
            all_data = np.array(all_data).reshape(-1, 1)
            self.scaler.fit(all_data)
        else:
            self.scaler = scaler
            self.mean = scaler.mean_[0]
            self.std = np.sqrt(scaler.var_[0])

        self.news_embeddings_original: Optional[torch.Tensor] = None
        self.news_embeddings_ver_primitive: Optional[torch.Tensor] = None
        # use_multimodal：是否启用文本/新闻模态；与「单路/双路嵌入」无关，双路仅由多模态+use_news_embedding 触发
        self._fnspid_dual_news = bool(self.use_multimodal and self.use_news_embedding)

        if self.use_news_embedding and not self.use_multimodal:
            raise ValueError(
                "FNSPID：单模态(use_multimodal=False)下不会加载新闻嵌入；请关闭 use_news_embedding，"
                "或与多模态模型联用时将 use_multimodal 设为 True。"
            )

        n_samples = len(self.data)
        if self._fnspid_dual_news:
            ko = f"{flag}_news_embed_file_original"
            kv = f"{flag}_news_embed_file_ver_primitive"
            path_o = _config_lookup(config, ko, None)
            path_v = _config_lookup(config, kv, None)
            if not path_o or not path_v:
                raise ValueError(
                    f"FNSPID 双嵌入模式需要同时配置 {ko} 与 {kv}（由 configurator 或 yaml 提供）"
                )
            self.news_embeddings_original = _load_split_news_tensor(
                dataset_path,
                path_o,
                flag,
                config,
                n_samples,
                config_key_for_tensor_name=ko,
                embed_key_config_suffix="original",
            )
            self.news_embeddings_ver_primitive = _load_split_news_tensor(
                dataset_path,
                path_v,
                flag,
                config,
                n_samples,
                config_key_for_tensor_name=kv,
                embed_key_config_suffix="ver_primitive",
            )
            if self.news_mode == "zero":
                self.news_embeddings_original = torch.zeros_like(self.news_embeddings_original)
                self.news_embeddings_ver_primitive = torch.zeros_like(self.news_embeddings_ver_primitive)

        self.samples = []
        for new_idx, sample in enumerate(self.data):
            hist_data = self._normalize_str(sample["historical_data"])
            if self.price_mode == "zero":
                hist_data = [0.0] * len(hist_data)
            gt_data = self._normalize_str(sample["ground_truth"])

            item = {
                "x": torch.tensor(hist_data, dtype=torch.float32),
                "y": torch.tensor(gt_data, dtype=torch.float32),
            }
            if self._fnspid_dual_news:
                item["news_embedding_original"] = self.news_embeddings_original[new_idx]
                item["news_embedding_ver_primitive"] = self.news_embeddings_ver_primitive[new_idx]
            self.samples.append(item)

    def _normalize_str(self, data_str):
        values = np.array(list(map(float, data_str.split(",")))).reshape(-1, 1)
        normalized = self.scaler.transform(values).flatten()
        return normalized.tolist()

    def get_scaler(self):
        return self.scaler

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        if self._fnspid_dual_news:
            return (
                sample["x"],
                sample["news_embedding_original"],
                sample["news_embedding_ver_primitive"],
                sample["y"],
            )
        return sample["x"], sample["y"]

    def inverse_transform(self, normalized_data):
        if isinstance(normalized_data, torch.Tensor):
            return normalized_data * self.std + self.mean
        return normalized_data * self.std + self.mean
