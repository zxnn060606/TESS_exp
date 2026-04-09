# coding: utf-8
# @email: enoche.chow@gmail.com

"""
MMTSF quick start 

"""
from cmath import inf
from logging import getLogger
from itertools import product
# from utils.dataset import MMTSFDataset,BitcoinDataset

from model_trainer.utils.logger import init_logger
from model_trainer.utils.configurator import Config, infer_default_model_from_dataset
from model_trainer.utils.utils import init_seed, get_model, dict2str
import platform
import os
import argparse
import importlib
import sys
from pathlib import Path
from itertools import product
from typing import Dict, Any

import torch

from tqdm import tqdm
from model_trainer.common.trainer import Trainer
from model_trainer.common.dataloader import data_loader
import warnings
warnings.filterwarnings("ignore")  # 忽略所有警告


def quick_start(dataset, config_dict=None, save_model=True, *, model=None):
    if config_dict is None:
        config_dict = {}
    effective_model = model or config_dict.get('model') or infer_default_model_from_dataset(dataset)
    config = Config(config_dict, effective_model, dataset)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info('██Server: \t' + platform.node())
    logger.info('██Dir: \t' + os.getcwd() + '\n')
    logger.info(config)

    # load data
    train_loader,valid_loader,test_loader = data_loader(config=config)
    
 
    ############ run model
    hyper_ret = []
    # val_metric = config['valid_metric']
    best_test_value = inf
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    if "seed" not in config['hyper_parameters']:
        config['hyper_parameters'] = ['seed'] + config['hyper_parameters']
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        # random seed reset
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k
        init_seed(config['seed'])

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))
        # wrap into dataloader
      
        


        # model loading and initialization
 
        nn_model = get_model(config['model'])(config).to(config['device'])
        nn_model = nn_model.float()

        # trainer loading and initialization
        trainer = Trainer(nn_model, config)
        
        # model training
        best_valid_score,best_test_upon_valid = trainer.fit(train_loader, valid_loader=valid_loader, test_loader=test_loader, saved=save_model)
        #########
                        
        hyper_ret.append((hyper_tuple,best_test_upon_valid))
      
        if best_test_upon_valid["MSE"] < best_test_value:
            best_test_value = best_test_upon_valid["MSE"]
            best_test_idx = idx
        idx += 1 

        logger.info('test result: {}'.format(dict2str(best_test_upon_valid)))
        logger.info('████Current BEST████:\nParameters: {}={},\n'
                    'Test: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], dict2str(hyper_ret[best_test_idx][1])))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k) in hyper_ret:
        logger.info('Parameters: {}={},\n best test: {}'.format(config['hyper_parameters'],
                                                                                  p, dict2str(k)))

    logger.info('\n\n█████████████ BEST ████████████████')
    logger.info('\tParameters: {}={},\nTest: {}\n\n'.format(config['hyper_parameters'],
                                                                   hyper_ret[best_test_idx][0],
                                                                   dict2str(hyper_ret[best_test_idx][1])))

