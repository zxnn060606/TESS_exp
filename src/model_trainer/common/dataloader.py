from model_trainer.common.dataset import (
    BitcoinDataset,
    ElectricityDataset,
    EnvironmentDataset,
    FnspidDataset,
)
import torch
from torch.utils.data import DataLoader
from logging import getLogger

from model_trainer.utils.embedding_checker import ensure_embeddings

def custom_collate_fn(batch):
    """将不同模态组合的样本统一组装成结构化批次。"""
    if not batch:
        raise ValueError("custom_collate_fn 收到空 batch，无法继续")

    x_list, y_list = [], []
   
    news_embed_original_list = []
    news_embed_ver_primitive_list = []

    for sample in batch:
        sample_len = len(sample)


        if sample_len == 4:
            x, second, third, y = sample
            # FNSPID 双嵌入：(x, emb_original, emb_ver_primitive, y)，两中间项均为 Tensor
            if torch.is_tensor(second) and torch.is_tensor(third):
                x_list.append(x)
                y_list.append(y)
                news_embed_original_list.append(second)
                news_embed_ver_primitive_list.append(third)
                continue
        elif sample_len == 2:
            x, y = sample
        else:
            raise ValueError(f"不支持的样本长度: {sample_len}")

        x_list.append(x)
        y_list.append(y)


    batch_dict = {
        'x': torch.stack(x_list),
        'y': torch.stack(y_list),
    }
    if news_embed_original_list:
        if len(news_embed_original_list) != len(x_list):
            raise ValueError("FNSPID 双embedding batch 长度不一致")
        batch_dict['news_embed_original'] = torch.stack(news_embed_original_list)
        batch_dict['news_embed_ver_primitive'] = torch.stack(news_embed_ver_primitive_list)
        batch_dict['news_embed'] = None
    return batch_dict

data_dict = {
    'Electricity':ElectricityDataset,
    'Bitcoin':BitcoinDataset,
    'Environment':EnvironmentDataset,
    'FNSPID':FnspidDataset,


}

def data_loader(config):
    logger = getLogger()
    config_dict = config.final_config_dict if hasattr(config, 'final_config_dict') else config
    ensure_embeddings(config_dict, logger=logger)
    data_class = data_dict[config['dataset']]
    train_dataset = data_class(config,flag="train")
    scaler = train_dataset.get_scaler()
    
    vali_dataset = data_class(config,flag="vali",scaler = scaler)
    test_dataset = data_class(config,flag="test",scaler = scaler)
    pin_memory = bool(config['use_gpu']) if 'use_gpu' in config else True
    train_loader = DataLoader(
                train_dataset,
                batch_size=config["batch_size"],
                shuffle=True,
                collate_fn=custom_collate_fn,
                num_workers=0,
                pin_memory=pin_memory
            )
    valid_loader = DataLoader(
            vali_dataset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate_fn
        )
    test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            collate_fn=custom_collate_fn
        )
    logger.info('\n====Current Dataset====\n'+str(config['dataset']))
    logger.info('\n====Training====\n' + str(len(train_dataset)))
    logger.info('\n====Validation====\n' + str(len(vali_dataset)))
    logger.info('\n====Testing====\n' + str(len(test_dataset)))
    return train_loader,valid_loader,test_loader
