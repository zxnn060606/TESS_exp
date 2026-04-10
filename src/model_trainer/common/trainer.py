import torch
from tqdm import tqdm
from time import time
from logging import getLogger
from typing import Any, Dict, List
import torch.optim as optim
from model_trainer.utils.evaluator import TemporalEvaluator
import numpy as np  
from model_trainer.utils.utils import early_stopping
import torch.nn as nn
from model_trainer.utils.utils import adjust_learning_rate
from model_trainer.utils.metrics import metric
from model_trainer.utils.artifact_manager import ArtifactManager
import os
import sys


class Trainer:
    def __init__(self, model,config):
        self.logger = getLogger()
        self.config = config
        
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.clip_grad_norm = config['clip_grad_norm']
        self.valid_metric = config['valid_metric']
        self.model = model
        self.device = config["device"]
        self.inverse = False
        
        self.weight_decay = 0.0
        if config['weight_decay'] is not None:
            wd = config['weight_decay']
            self.weight_decay = eval(wd) if isinstance(wd, str) else wd
        self.req_training = config['req_training']
        self.start_epoch = 0
        self.cur_step = 0
        
        self.best_valid_score = float('inf')
        self.train_loss_dict = dict()
        self.best_state_dict = None
        self.best_metrics = None
    

        self.optimizer = self._build_optimizer()

        self.evaluator = TemporalEvaluator(config)
        self.patience = config['patience']
        self.path = './checkpoints/'
        self.use_multimodal = bool(config['use_multimodal']) if 'use_multimodal' in config else False
        self.use_primitive = bool(config.get('use_primitive', False))
        self.artifacts = ArtifactManager(config)

    def _pick_news_feat(self, news_original, news_ver_primitive, single_news_embed):
        """按 use_primitive 选一路"""
        if isinstance(news_original, torch.Tensor) and isinstance(news_ver_primitive, torch.Tensor):
            return news_ver_primitive if self.use_primitive else news_original
        return single_news_embed

    def _prepare_batch(self, batch_data):
        """兼容新旧 collate 输出，统一整理成字典。"""
        if isinstance(batch_data, dict):

            batch_x = batch_data['x'].float().to(self.device)
            batch_y = batch_data['y'].float().to(self.device)

            news_o = batch_data.get('news_embed_original')
            news_v = batch_data.get('news_embed_ver_primitive')
            if isinstance(news_o, torch.Tensor) and isinstance(news_v, torch.Tensor):
                news_o = news_o.float().to(self.device)
                news_v = news_v.float().to(self.device)
                news_embed = self._pick_news_feat(news_o, news_v, None)
                return {
                    'x': batch_x,
                    'y': batch_y,
                    'news_embed_original': news_o,
                    'news_embed_ver_primitive': news_v,
                    'news_embed': news_embed,
                    'batch_size': batch_x.size(0),
                }
            else:
                raise ValueError("无完整的embedding")



    def _compute_outputs(self, batch: Dict[str, torch.Tensor], mode: str):
        batch_x = batch['x']
        batch_y = batch['y']
        # 调模型前再次按 use_primitive 从双路中明确选型（与 _prepare_batch 一致，避免仅依赖隐式缓存字段）
        news_feat = self._pick_news_feat(
            batch.get('news_embed_original'),
            batch.get('news_embed_ver_primitive'),
            batch.get('news_embed'),
        )

        if self.use_multimodal:
            if news_feat is None:
                raise ValueError(
                    "多模态训练需要新闻 embedding："
                    "FNSPID 嵌入需 batch 含 news_embed_original 与 news_embed_ver_primitive；"
                )
            if 'flag' in self.model.forward.__code__.co_varnames:
                outputs = self.model(batch_x, news_feat, flag=mode)
            else:
                outputs = self.model(batch_x, news_feat)
        else:
            outputs = self.model(batch_x)

        return outputs


    def _config_snapshot(self) -> Dict:
        snapshot = {}
        for key, value in self.config.final_config_dict.items():
            if isinstance(value, torch.device):
                snapshot[key] = str(value)
            else:
                snapshot[key] = value
        return snapshot

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = optim.Adagrad(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            self.logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
   
    def train_epoch(self, train_loader, epoch_idx, loss_func=None):
        """单个训练步骤"""
        if not self.req_training:
            return 0.0

        train_loss = []
        running_avg_loss = 0.0
        with tqdm(
            train_loader,
            desc=f"Epoch {epoch_idx + 1}/{self.epochs}",
            leave=False,
        ) as progress:
            for batch_idx, batch_data in enumerate(progress):
                self.optimizer.zero_grad()

                batch = self._prepare_batch(batch_data)
                outputs = self._compute_outputs(batch, mode='train')
                batch_y = batch['y']

                if hasattr(self.model, 'calculate_loss'):
                    loss = self.model.calculate_loss(batch_y)
                else:
                    loss = loss_func(outputs, batch_y)

                loss_value = loss.item()
                train_loss.append(loss_value)
                loss.backward()
                self.optimizer.step()

                running_avg_loss = (running_avg_loss * batch_idx + loss_value) / (batch_idx + 1)
                progress.set_postfix(loss=f"{loss_value:.4f}", avg=f"{running_avg_loss:.4f}")

        train_loss = np.average(train_loss) if train_loss else 0.0

        return train_loss

    def vali(self, vali_loader, loss_func):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            with tqdm(vali_loader, desc="Validation", leave=False) as progress:
                for batch_idx, batch_data in enumerate(progress):
                    batch = self._prepare_batch(batch_data)
                    outputs = self._compute_outputs(batch, mode='test')
                    batch_y = batch['y']
                    pred = outputs.detach().cpu()
                    true = batch_y.detach().cpu()

                    loss = loss_func(pred, true)

                    total_loss.append(float(loss.item()))

        total_loss = np.average(total_loss) if total_loss else 0.0
        self.model.train()
        return total_loss

    def _evaluate_split(self, loader, split: str, return_raw: bool = False):
        preds = []
        trues = []
        dataset = loader.dataset

        self.model.eval()
        with torch.no_grad():
            with tqdm(loader, desc=f"{split.capitalize()} Eval", leave=False) as progress:
                for batch_idx, batch_data in enumerate(progress):
                    batch = self._prepare_batch(batch_data)
                    outputs = self._compute_outputs(batch, mode='test')
                    batch_y = batch['y']
                    outputs = outputs.float().detach().cpu().numpy()
                    batch_y_np = batch_y.float().detach().cpu().numpy()
                    if self.inverse:
                        pred = dataset.inverse_transform(outputs)
                        gt = dataset.inverse_transform(batch_y_np)
                    else:
                        pred = outputs
                        gt = batch_y_np

                    preds.append(pred)
                    trues.append(gt)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print(f'{split} shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        metrics_dict = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape,
            "MSPE": mspe,
        }

        if return_raw:
            return metrics_dict, preds, trues
        return metrics_dict



    @staticmethod
    def _to_serializable(value: Any):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, list):
            return [Trainer._to_serializable(v) for v in value]
        if isinstance(value, tuple):
            return [Trainer._to_serializable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    

    def test(self, test_loader, return_raw: bool = False):
        """评估模型，必要时返回逐样本预测用于滞后分析。"""
        return self._evaluate_split(test_loader, 'test', return_raw=return_raw)

    

    def fit(self, train_loader, valid_loader, test_loader=None,saved = False,verbose = True):
        """执行完整训练流程"""

        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        for epoch_idx in range(self.start_epoch, self.epochs):
                # train
                self.model.train()
                training_start_time = time()
                loss_func = nn.MSELoss() 
                
                train_loss = self.train_epoch(train_loader, epoch_idx,loss_func=loss_func)
                self.artifacts.write_epoch_metrics(epoch_idx, {'loss': float(train_loss)}, 'train')
                
                valid_start_time = time()
                vali_score = self.vali(vali_loader=valid_loader,loss_func=loss_func)
                self.artifacts.write_epoch_metrics(epoch_idx, {'loss': float(vali_score)}, 'vali')
                valid_end_time = time()
                valid_score_output = "epoch %d evaluating [time: %.2fs, valid_score: %f]" % \
                                        (epoch_idx, valid_end_time - valid_start_time, vali_score)
                

                self.best_valid_score,self.cur_step,stop_flag, update_flag = early_stopping(vali_score,self.best_valid_score,self.cur_step,max_step=self.patience)
                
                adjust_learning_rate(self.optimizer, epoch_idx + 1, self.learning_rate)
                metrics_dict = self.test(test_loader=test_loader)
                self.logger.info(
                    '\n Current Epoch {} Test Result: MAE: {}, MSE: {}, RMSE: {}, MAPE: {}, MSPE: {}'.format(
                        epoch_idx,
                        metrics_dict['MAE'],
                        metrics_dict['MSE'],
                        metrics_dict['RMSE'],
                        metrics_dict['MAPE'],
                        metrics_dict['MSPE'],
                    )
                )
                
                
                if update_flag:
                    update_output = '██ ' + self.config['model'] + '--Best validation results updated!!!'
                    self.logger.info(update_output)
                    self.best_test_upon_valid = metrics_dict
                    self.best_state_dict = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                    self.best_metrics = metrics_dict
                    self.artifacts.save_best_model(self.model)
                    self.artifacts.save_config_snapshot(
                        self._config_snapshot(),
                        best_epoch=epoch_idx + 1,
                        valid_score=vali_score,
                        test_metrics=metrics_dict,
                    )
                    self.artifacts.write_epoch_metrics(epoch_idx, metrics_dict, 'test')
                if stop_flag:
                    stop_output = '+++++Finished training, best eval result in epoch %d' % \
                                    (epoch_idx - self.cur_step * self.eval_step)
                    self.logger.info(stop_output)
                    break

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        if self.artifacts.export_samples:
            split_loaders = [
                ('train', train_loader),
                ('vali', valid_loader),
                ('test', test_loader),
            ]
            for split, loader in split_loaders:
                if loader is None:
                    continue
                metrics_dict, preds, trues = self._evaluate_split(loader, split, return_raw=True)
                self.artifacts.write_sample_scores(split, preds, trues)
                self.artifacts.write_epoch_metrics('best', metrics_dict, split)

        if self.best_metrics is not None:
            self.artifacts.manifest['best_metrics'] = self.artifacts._sanitize_metrics(self.best_metrics)
        self.artifacts.write_manifest()

        if self.best_test_upon_valid is None and 'metrics_dict' in locals():
            self.best_test_upon_valid = metrics_dict

        return self.best_valid_score,self.best_test_upon_valid

                


    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses):
        train_loss_output = 'epoch %d training [time: %.2fs, ' % (epoch_idx, e_time - s_time)
        if isinstance(losses, tuple):
            train_loss_output = ', '.join('train_loss%d: %.4f' % (idx + 1, loss) for idx, loss in enumerate(losses))
        else:
            train_loss_output += 'train loss: %.4f' % losses
        return train_loss_output + ']'
