import os
import numpy as np
import argparse
import time
import torch

from datasets import ClickDataset
from behavior_aggregators import AggregatorWeights
from models import MatrixFactorization
from cf_config import CFConfig
from train import Engine
import metrics

import utils

if __name__ == "__main__":
    print('this is main ...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./benchmarks/AmazonBooks/MF_CCL/configs/config0.yaml', 
                        help='The config file for para config.')
    args = parser.parse_args()

    config_path = args.config
    config_dic = utils.load_config(config_path)
    dataset_config = config_dic['dataset_config']
    model_config = config_dic['model_config']
    print(model_config)

    cf_config = CFConfig(emb_dim=model_config['embedding_dim'], num_negs=model_config['num_negs'], max_his=model_config['max_his'], neg_sampler=model_config['neg_sampler'], 
        tile_size=model_config['tile_size'], refresh_interval=model_config['refresh_interval'], l2=model_config['embedding_regularizer'], clip_val=model_config['clip_val'], 
        l_r=model_config['learning_rate'])

    train_file = os.path.join(dataset_config['data_dir'], dataset_config['train_data'])
    train_data = ClickDataset(train_file, separator=dataset_config['separator'], config=cf_config)

    cf_config.init_c_instance()

    test_file = os.path.join(dataset_config['data_dir'], dataset_config['test_data'])
    test_data = ClickDataset(test_file, separator=dataset_config['separator'], config=cf_config)

    aggregator_weights = AggregatorWeights(cf_config)
    model = MatrixFactorization(cf_config)
    model.init_c_instance(cf_config)
    engine = Engine(train_data, aggregator_weights, model, cf_config)

    for epoch in range(model_config['epochs']):
        start_time = time.time()

        epoch_loss = engine.train_one_epoch()

        epoch_time = time.time() - start_time
        print(f'epoch: {epoch}; loss: {epoch_loss}; epoch_time: {epoch_time}')

        if epoch > 0 and epoch % model_config['eval_interval'] == 0:
            print('--- Start evaluation ---')
            model.eval()
            with torch.no_grad():
                eva_metrics = ['Recall(k=20)']
                metrics.evaluate_metrics(model, train_data, test_data, eva_metrics)

