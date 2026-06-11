from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import yaml
import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np

from model import FBNETGEN, GNNPredictor, SeqenceModel, BrainNetCNN
from model.model import FCNet
from model.model3 import SGC, SGC2
from model.model4 import NETGEN, DGTransNet, NETGEN2
from train import FCNetTrain
from train2 import BiLevelTrain, SeqTrain, GNNTrain, BrainCNNTrain #, BasicTrain
from train3 import  BasicTrain
# from train4 import  SGCTrain, BasicTrain
from datetime import datetime
from dataloader import init_dataloader


# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu_available = torch.cuda.is_available()
# print(f"GPU 可用：{gpu_available}")


def main(args):
    with open(args.config_filename) as f:
        config = yaml.load(f, Loader=yaml.Loader)

        dataloaders, node_size, node_feature_size, timeseries_size = \
            init_dataloader(config['data'])

        config['train']["seq_len"] = timeseries_size
        config['train']["node_size"] = node_size
#
        if config['model']['type'] == 'seq':
            model = SeqenceModel(config['model'], node_size, timeseries_size)
            use_train = SeqTrain

        elif config['model']['type'] == 'gnn':
            model = GNNPredictor(node_feature_size, node_size)
            use_train = GNNTrain

        elif config['model']['type'] == 'FBNETGEN':
            model = FBNETGEN(config['model'], node_size,
                             node_feature_size, timeseries_size)
            use_train = BasicTrain

        elif config['model']['type'] == 'brainnetcnn':

            model = BrainNetCNN(node_size)

            use_train = BrainCNNTrain

        # elif config['model']['type'] == 'SGC2':     #
        #     model = SGC2(config['model'], node_size)   #
        #     use_train = SGCTrain    #

        elif config['model']['type'] == 'fcnet':
            model = FCNet(node_size, timeseries_size)
            model = model.to(device)  # 确保模型在正确的设备上
            use_train = FCNetTrain

        elif config['model']['type'] == 'NETGEN':
            model = NETGEN(config['model'], node_size,node_feature_size, timeseries_size)
            use_train = BasicTrain
        elif config['model']['type'] == 'NETGEN2':
            model = NETGEN2(config['model'], node_size,node_feature_size, timeseries_size)
            use_train = BasicTrain

        elif config['model']['type'] == 'DGT':
            model = DGTransNet(config['model'], node_size,node_feature_size, timeseries_size)
            use_train = BasicTrain

        if config['train']['method'] == 'BiLevelTrain' and \
                config['model']['type'] == 'FBNETGEN':
            parameters = {
                'lr': config['train']['lr'],
                'weight_decay': config['train']['weight_decay'],
                'params': [
                    {'params': model.extract.parameters()},
                    {'params': model.emb2graph.parameters()}
                ]
            }

            optimizer1 = torch.optim.Adam(**parameters)

            optimizer2 = torch.optim.Adam(model.predictor.parameters(),
                                          lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
            opts = (optimizer1, optimizer2)
            use_train = BiLevelTrain

        else:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['train']['lr'],
                weight_decay=config['train']['weight_decay'])
            opts = (optimizer,)

        loss_name = 'loss'
        if config['train']["group_loss"]:
            loss_name = f"{loss_name}_group_loss"
        if config['train']["sparsity_loss"]:
            loss_name = f"{loss_name}_sparsity_loss"

        # now = datetime.now()
        #
        # date_time = now.strftime("%m-%d-%H-%M-%S")

        extractor_type = config['model']['extractor_type'] if 'extractor_type' in config['model'] else "none"
        embedding_size = config['model']['embedding_size'] if 'embedding_size' in config['model'] else "none"
        window_size = config['model']['window_size'] if 'window_size' in config['model'] else "none"

        save_folder_name = Path(config['train']['log_folder'])/Path(
            # date_time +
            f"_{config['data']['dataset']}_{config['model']['type']}_{config['train']['method']}" )

        # # 将dataloader中的数据移动到 GPU
        # for phase in dataloaders:
        #     for batch in dataloaders[phase]:
        #         # 这里假设dataloader中的数据是一个字典，包含'text'和'label'，将它们一起转移到GPU
        #         for key in batch:
        #             batch[key] = batch[key].to(device)

        train_process = use_train(
            config['train'], model, opts, dataloaders, save_folder_name)

        train_process.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filename', default='setting/abide_fbnetgen.yaml', type=str,
                        help='Configuration filename for training the model.')
    parser.add_argument('--repeat_time', default=50, type=int)
    args = parser.parse_args()
    for i in range(args.repeat_time):
        main(args)
        # torch.cuda.set_device(0)
        # seed = 12344
        # random.seed(seed)
        # np.random.seed(seed)
        # if torch.cuda.is_available():
        #     torch.cuda.manual_seed_all(seed)
        #
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        #
        # cudnn.deterministic = True
        #
        # main(args)
