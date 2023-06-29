# update in 2023/6/28 21：15 by liujunyan
import argparse
import os

import ast
import pickle
import sys
import torch
from torch.utils.data import WeightedRandomSampler

from model.cav_mae import CAVMAEFT
from configs.nsconfig import cavmaeconfig, trainconfig
from dataset.dataloader import get_dataloader, get_testloader
import numpy as np
import warnings
import json
from sklearn import metrics
from train.train import train
from model.cnntest import get_resnet50
from torchsummary import summary
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(100)

    audio_model = CAVMAEFT(label_dim=cavmaeconfig['n_class'],
                           modality_specific_depth=cavmaeconfig['modality_specific_depth'])
    # for k, v in audio_model.state_dict().items():
    #     print(k)
    mdl_weight = torch.load(cavmaeconfig['pretrain_path'])
    if not isinstance(audio_model, torch.nn.DataParallel):
        audio_model = torch.nn.DataParallel(audio_model)
    miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    print('now load cav-mae pretrained weights from ', cavmaeconfig['pretrain_path'])
    print(miss, unexpected)

    train_loader, val_loader = get_dataloader('mean')
    summary(audio_model, input_size=[(1024, 128), (3, 224, 224)], device='cpu')
    train(audio_model, train_loader, val_loader, start_epoch=0)


