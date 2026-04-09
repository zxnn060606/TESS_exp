import re
import os
import yaml
import torch
from logging import getLogger
import ipdb
from typing import Dict

from model_trainer.utils.dataset_registry import DatasetRegistry, DatasetRegistryError


def infer_default_model_from_dataset(dataset: str) -> str:
    """仅读 overall + dataset yaml 中的 use_multimodal，决定默认模型 yaml / 类名（先于 model yaml 合并）。"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cur_dir = os.path.join(base_dir, 'configs')
    merged: Dict = {}
    for path in (
        os.path.join(cur_dir, 'overall.yaml'),
        os.path.join(cur_dir, 'dataset', f'{dataset}.yaml'),
    ):
        if not os.path.isfile(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f.read()) or {}
        if isinstance(data, dict):
            merged.update(data)
    return 'MultiModal_Baseline' if bool(merged.get('use_multimodal')) else 'UniModal_Baseline'


class Config:
    def __init__(self, config_dict =None, model = None, dataset = None):
        """
       配置管理,方便超参数搜索
        """
        if config_dict is None:
            config_dict = {}
        config_dict['model'] = model
        config_dict['dataset'] = dataset
        # model type
        self.final_config_dict = self._load_dataset_model_config(config_dict)
        # config in cmd and main.py are latest
        self.final_config_dict.update(config_dict)
        self._apply_dataset_registry()
        self._set_default_parameters()
        self._init_device()

    def _load_dataset_model_config(self, config_dict):
        file_config_dict = dict()
        file_list = []
        # get dataset and model files
        # Locate configs directory relative to this file to avoid CWD issues
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/model_trainer
        cur_dir = os.path.join(base_dir, 'configs')
        file_list.append(os.path.join(cur_dir, "overall.yaml"))
        file_list.append(os.path.join(cur_dir, "dataset", "{}.yaml".format(config_dict['dataset'])))
        file_list.append(os.path.join(cur_dir, "model", "{}.yaml".format(config_dict['model'])))

        hyper_parameters = []
        for file in file_list:
            if os.path.isfile(file):
                with open(file, 'r', encoding='utf-8') as f:
                    fdata = yaml.load(f.read(), Loader=self._build_yaml_loader())
                    if fdata.get('hyper_parameters'):
                        hyper_parameters.extend(fdata['hyper_parameters'])
                    file_config_dict.update(fdata)
                    
        file_config_dict['hyper_parameters'] = hyper_parameters
        return file_config_dict
    
    def _apply_fnspid_dual_news_embed_keys(self) -> None:
        """FNSPID 多模态+新闻嵌入：写入两套 *_news_embed_file_* 键（yaml 已填的项不覆盖）。"""
        if not (
            str(self.final_config_dict.get('dataset', '')).upper() == 'FNSPID'
            and bool(self.final_config_dict.get('use_multimodal'))
            and bool(self.final_config_dict.get('use_news_embedding'))
        ):
            return

        orig_alias = self.final_config_dict.get('fnspid_news_embed_alias_original', 'FNSPID/ver_camf')
        prim_alias = self.final_config_dict.get('fnspid_news_embed_alias_ver_primitive', 'FNSPID/ver_primitive')
        overrides: Dict = self.final_config_dict.get('dataset_overrides') or None

        def _fill_from_alias(alias: str, suffix: str) -> None:
            try:
                info = DatasetRegistry.get(alias, overrides=overrides)
            except DatasetRegistryError as exc:
                raise DatasetRegistryError(
                    f"Failed to resolve FNSPID news embed alias '{alias}' ({suffix}): {exc}"
                ) from exc
            news_embedding = (info.get('embeddings') or {}).get('news')
            if not news_embedding:
                raise DatasetRegistryError(f"Alias '{alias}' 未定义 news embeddings（{suffix}）")
            base_path = news_embedding.get('path')
            split_map = news_embedding.get('splits', {}) or {}
            if not base_path:
                raise DatasetRegistryError(f"Alias '{alias}' 缺少 embedding path（{suffix}）")
            for split_name, key_stub in [['train', 'train'], ['vali', 'vali'], ['test', 'test']]:
                if split_name not in split_map:
                    continue
                embed_key = f"{key_stub}_news_embed_file_{suffix}"
                if self.final_config_dict.get(embed_key):
                    continue
                embed_value = split_map[split_name]
                if embed_value is not None:
                    self.final_config_dict[embed_key] = f"{base_path}::{embed_value}"
            path_key = f'news_embedding_path_{suffix}'
            if not self.final_config_dict.get(path_key):
                self.final_config_dict[path_key] = base_path

        _fill_from_alias(orig_alias, 'original')
        _fill_from_alias(prim_alias, 'ver_primitive')

    def _apply_dataset_registry(self):
        dataset = self.final_config_dict.get('dataset')
        if str(dataset).upper() == 'FNSPID':
            # 划分 JSON 仅由 yaml 提供，不从 registry 覆盖 train/vali/test
            self._apply_fnspid_dual_news_embed_keys()
            self._apply_gt_embedding_alias_block()
            return

        alias = self.final_config_dict.get('dataset_alias')
        dataset_version = self.final_config_dict.get('dataset_version')
        if not alias and dataset:
            if dataset_version:
                alias = f"{dataset}/{dataset_version}"
        if not alias:
            self._apply_gt_embedding_alias_block()
            return

        overrides: Dict = self.final_config_dict.get('dataset_overrides') or None
        try:
            registry_info = DatasetRegistry.get(alias, overrides=overrides)
        except DatasetRegistryError as exc:
            raise DatasetRegistryError(
                f"Failed to resolve dataset alias '{alias}': {exc}"
            ) from exc

        self.final_config_dict['dataset_alias'] = registry_info.get('alias', alias)
        if registry_info.get('version'):
            self.final_config_dict['dataset_version'] = registry_info['version']

        dataset_root = registry_info.get('root')
        if dataset_root:
            self.final_config_dict['dataset_root'] = dataset_root
            # data_path 保持兼容旧逻辑
            self.final_config_dict['data_path'] = dataset_root

        splits = registry_info.get('splits', {})
        split_key_map = {
            'train': 'train_file',
            'vali': 'vali_file',
            'valid': 'vali_file',
            'test': 'test_file'
        }
        for split_name, config_key in split_key_map.items():
            if split_name in splits:
                self.final_config_dict[config_key] = splits[split_name]

        embeddings = registry_info.get('embeddings', {})
        news_embedding = embeddings.get('news')
        if news_embedding:
            base_path = news_embedding.get('path')
            split_map = news_embedding.get('splits', {}) or {}
            if base_path:
                for split_name, key_stub in [['train', 'train'], ['vali', 'vali'], ['test', 'test']]:
                    if split_name in split_map:
                        embed_key = f"{key_stub}_news_embed_file"
                        embed_value = news_embedding['splits'][split_name]
                        if embed_value is not None:
                            self.final_config_dict[embed_key] = f"{base_path}::{embed_value}"
            self.final_config_dict['news_embedding_path'] = base_path

        # 保留嵌入注册信息，供后续组件使用
        if embeddings:
            self.final_config_dict['embedding_registry'] = embeddings

        self._apply_gt_embedding_alias_block()

    def _apply_gt_embedding_alias_block(self) -> None:
        """处理 GT embedding（如果配置了 gt_embedding_alias）。"""
        gt_embedding_alias = self.final_config_dict.get('gt_embedding_alias')
        if gt_embedding_alias:
            try:
                gt_registry_info = DatasetRegistry.get(gt_embedding_alias)
                gt_embeddings = gt_registry_info.get('embeddings', {})
                gt_news_embedding = gt_embeddings.get('news')
                if gt_news_embedding:
                    gt_base_path = gt_news_embedding.get('path')
                    gt_split_map = gt_news_embedding.get('splits', {}) or {}
                    if gt_base_path:
                        # 获取 GT embedding 的数据集根目录
                        gt_dataset_root = gt_registry_info.get('root')
                        if gt_dataset_root:
                            # 构建完整的 GT embedding 路径（相对于 GT 数据集根目录）
                            for split_name, key_stub in [['train', 'train'], ['vali', 'vali'], ['test', 'test']]:
                                if split_name in gt_split_map:
                                    gt_embed_key = f"{key_stub}_gt_embed_file"
                                    gt_embed_value = gt_news_embedding['splits'][split_name]
                                    if gt_embed_value is not None:
                                        # 格式：路径::键名，例如：embedding_qwen/all_token_embeddings.pt::train
                                        self.final_config_dict[gt_embed_key] = f"{gt_base_path}::{gt_embed_value}"
                            self.final_config_dict['gt_embedding_path'] = gt_base_path
                            self.final_config_dict['gt_dataset_root'] = gt_dataset_root
                else:
                    raise DatasetRegistryError(f"GT embedding alias '{gt_embedding_alias}' 未定义 news embeddings")
            except DatasetRegistryError as exc:
                raise DatasetRegistryError(
                    f"Failed to resolve GT embedding dataset alias '{gt_embedding_alias}': {exc}"
                ) from exc
    
    
    def _set_default_parameters(self):
        smaller_metric = ['rmse', 'mae', 'mse']
        valid_metric = self.final_config_dict['valid_metric']
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric in smaller_metric else True
        # if seed not in hyper_parameters, then add
        if "seed" not in self.final_config_dict['hyper_parameters']:
            self.final_config_dict['hyper_parameters'] += ['seed']
        if 'export_sample_metrics' not in self.final_config_dict:
            self.final_config_dict['export_sample_metrics'] = True
        if 'use_primitive' not in self.final_config_dict:
            self.final_config_dict['use_primitive'] = False

    def _init_device(self):
        use_gpu = self.final_config_dict['use_gpu']
        gpu_id = self.final_config_dict.get('gpu_id', 0)  # 获取 GPU ID，默认为 0

        if use_gpu and torch.cuda.is_available():
            self.final_config_dict['device'] = torch.device(f"cuda:{gpu_id}")  # 绑定到指定 GPU
        else:
            self.final_config_dict['device'] = torch.device("cpu") 
            
    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config_dict[key] = value

    def __getitem__(self, item):
        if item in self.final_config_dict:
            return self.final_config_dict[item]
        else:
            return None

    def get(self, key, default=None):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        value = self.final_config_dict.get(key, default)
        return value if value is not None else default

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config_dict
    
    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
        return loader
  
    def __str__(self):
        args_info = '\n'
        args_info += '\n'.join(["{}={}".format(arg, value) for arg, value in self.final_config_dict.items()])
        args_info += '\n\n'
        return args_info
    def __repr__(self):
        return self.__str__()
