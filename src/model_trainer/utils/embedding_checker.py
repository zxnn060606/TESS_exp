import json
import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from model_trainer.utils.dataset_registry import DatasetRegistry, DatasetRegistryError


@dataclass
class EmbeddingSpec:
    path: str
    splits: Dict[str, str]


def _ensure_fnspid_dual_embeddings(
    config: Dict,
    dataset_root: str,
    *,
    logger=None,
) -> None:
    """FNSPID 多模态：校验 ver_camf 与 ver_primitive 两路嵌入文件（与 yaml 划分行数对齐）；缺失则报错。"""
    orig_alias = config.get('fnspid_news_embed_alias_original', 'FNSPID/ver_camf')
    prim_alias = config.get('fnspid_news_embed_alias_ver_primitive', 'FNSPID/ver_primitive')
    config_splits = {
        'train': config.get('train_file'),
        'vali': config.get('vali_file'),
        'test': config.get('test_file'),
    }

    for alias in (orig_alias, prim_alias):
        try:
            registry_info = DatasetRegistry.get(alias)
        except DatasetRegistryError as exc:
            raise RuntimeError(f'failed to resolve FNSPID embedding alias {alias}: {exc}') from exc

        embeddings = registry_info.get('embeddings', {})
        news_spec_dict = embeddings.get('news')
        if not news_spec_dict:
            raise RuntimeError(f'alias {alias} does not declare news embeddings in registry')

        embed_spec = EmbeddingSpec(
            path=news_spec_dict.get('path', ''),
            splits=news_spec_dict.get('splits', {})
        )
        if not embed_spec.path:
            raise RuntimeError(f'alias {alias} missing embedding path definition')

        embed_abs_path = _make_abs_path(dataset_root, embed_spec.path)

        if not os.path.isfile(embed_abs_path):
            raise FileNotFoundError(
                f'Required news embedding file is missing: {embed_abs_path} '
                f'(registry alias={alias!r}, relative path={embed_spec.path!r}). '
                'Generate it with scripts/generate_qwen_embeddings.py or fix dataset_root / index.yaml paths.'
            )

        _validate_embedding_file(
            embed_abs_path,
            embed_spec,
            dataset_root,
            config_splits,
            logger=logger,
        )


def ensure_embeddings(config: Dict, *, logger=None) -> None:
    """校验所需嵌入文件存在且合法；不存在则直接报错（不自动调用生成脚本）。"""
    if config.get('legacy_loader'):
        return
    if not config.get('requires_news_embedding'):
        return

    dataset_root = config.get('dataset_root') or config.get('data_path')
    if not dataset_root:
        raise RuntimeError('dataset_root/data_path is required to locate embeddings')

    ds = str(config.get('dataset', '')).upper()
    if ds == 'FNSPID' and config.get('use_multimodal') and config.get('use_news_embedding'):
        _ensure_fnspid_dual_embeddings(config, dataset_root, logger=logger)
        return

    dataset_alias = config.get('dataset_alias')
    dataset = config.get('dataset')
    dataset_version = config.get('dataset_version')
    if not dataset_alias and dataset and dataset_version:
        dataset_alias = f'{dataset}/{dataset_version}'
    if not dataset_alias:
        raise RuntimeError(
            'requires_news_embedding=True：非 FNSPID 双嵌入时请配置 dataset_alias，或同时提供 dataset 与 dataset_version'
        )

    try:
        registry_info = DatasetRegistry.get(dataset_alias)
    except DatasetRegistryError as exc:
        raise RuntimeError(f'failed to resolve dataset alias {dataset_alias}: {exc}') from exc

    embeddings = registry_info.get('embeddings', {})
    news_spec_dict = embeddings.get('news')
    if not news_spec_dict:
        raise RuntimeError(f'alias {dataset_alias} does not declare news embeddings in registry')

    embed_spec = EmbeddingSpec(
        path=news_spec_dict.get('path', ''),
        splits=news_spec_dict.get('splits', {})
    )
    if not embed_spec.path:
        raise RuntimeError(f'alias {dataset_alias} missing embedding path definition')

    embed_abs_path = _make_abs_path(dataset_root, embed_spec.path)

    if not os.path.isfile(embed_abs_path):
        raise FileNotFoundError(
            f'Required news embedding file is missing: {embed_abs_path} '
            f'(alias={dataset_alias!r}, relative path={embed_spec.path!r}). '
            'Generate it with scripts/generate_qwen_embeddings.py or fix paths in configs/dataset/index.yaml.'
        )

    _validate_embedding_file(
        embed_abs_path,
        embed_spec,
        dataset_root,
        registry_info.get('splits', {}),
        logger=logger,
    )


def _validate_embedding_file(
    embed_abs_path: str,
    embed_spec: EmbeddingSpec,
    dataset_root: str,
    splits: Dict[str, str],
    *,
    logger=None,
) -> None:
    if logger:
        logger.info('Validating embedding file: %s', embed_abs_path)

    tensor_dict = torch.load(embed_abs_path, map_location='cpu')
    if not isinstance(tensor_dict, dict):
        raise ValueError(f'Embedding file {embed_abs_path} should contain a dict of tensors')

    expected_split_lengths = _load_split_lengths(dataset_root, splits)

    for split, key in embed_spec.splits.items():
        if key not in tensor_dict:
            raise KeyError(f'Embedding file missing key {key} for split {split}')

        split_data = tensor_dict[key]
        if not isinstance(split_data, dict):
            # 兼容旧格式：直接是tensor
            split_data = {'embeddings': split_data}

        # 验证embeddings
        if 'embeddings' not in split_data:
            raise KeyError(f'Embedding split {key} missing "embeddings" field')
        embeddings = split_data['embeddings']
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings)
        if not torch.is_tensor(embeddings):
            raise TypeError(f'Embeddings in {key} is not a tensor/ndarray')
        if embeddings.dim() not in [2, 3]:  # 支持2D(句子级)和3D(token级)
            raise ValueError(f'Embeddings in {key} must be 2D or 3D, got shape {tuple(embeddings.shape)}')
        if not torch.isfinite(embeddings).all():
            raise ValueError(f'Embeddings in {key} contains NaN/Inf values')
        if embeddings.dim() == 2 and (embeddings.norm(dim=1) == 0).any():
            raise ValueError(f'Embeddings in {key} contains zero vectors')
        elif embeddings.dim() == 3 and (embeddings.norm(dim=-1) == 0).any():
            # 对于3D，检查token级别的零向量
            zero_tokens = (embeddings.norm(dim=-1) == 0)
            if zero_tokens.any() and not zero_tokens.all() and logger:
                logger.warning(f'Embeddings in {key} contains some zero token vectors')

        expected_len = expected_split_lengths.get(split)
        if expected_len is not None and embeddings.shape[0] != expected_len:
            raise ValueError(
                f'Embeddings in {key} length {embeddings.shape[0]} mismatches dataset split {split} length {expected_len}'
            )

        # 验证attention mask (如果存在)
        if 'attention_mask' in split_data:
            mask = split_data['attention_mask']
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            if not torch.is_tensor(mask):
                raise TypeError(f'Attention mask in {key} is not a tensor/ndarray')
            if mask.shape[0] != embeddings.shape[0]:
                raise ValueError(f'Attention mask length {mask.shape[0]} mismatches embeddings length {embeddings.shape[0]}')
            if embeddings.dim() == 3 and mask.dim() == 2:
                if mask.shape[1] != embeddings.shape[1]:
                    raise ValueError(f'Attention mask seq_len {mask.shape[1]} mismatches embeddings seq_len {embeddings.shape[1]}')
            if mask.dtype not in [torch.bool, torch.int, torch.long] and logger:
                logger.warning(f'Attention mask in {key} should be boolean or integer type, got {mask.dtype}')
        elif embeddings.dim() == 3 and logger:
            logger.warning(f'3D embeddings in {key} found but no attention_mask provided - padding tokens may not be properly handled')


def _load_split_lengths(dataset_root: str, splits: Dict[str, str]) -> Dict[str, int]:
    lengths: Dict[str, int] = {}
    for split, rel_path in splits.items():
        if not rel_path:
            continue
        abs_path = _make_abs_path(dataset_root, rel_path)
        if not os.path.isfile(abs_path):
            continue
        with open(abs_path, 'r', encoding='utf-8') as f:
            records = json.load(f)
        lengths[split] = len(records)
    return lengths


def _make_abs_path(root: str, rel_path: str) -> str:
    rel = rel_path.lstrip('/')
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.abspath(os.path.join(root, rel))
