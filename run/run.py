# update in 2023/6/28 21：15 by liujunyan
import argparse
import os

import ast
import pickle
import sys
import torch
import torchviz
from torch.utils.data import WeightedRandomSampler
from model.vocalist import SyncTransformer, count_parameters
from model.cav_mae import CAVMAEFT
from configs.nsconfig import cavmaeconfig, trainconfig
from dataset.dataloader import get_dataloader, get_testloader
import numpy as np
import warnings
import json
from sklearn import metrics
import train.train_vocalist
from model.cnntest import get_resnet50
from torchsummary import summary
import random
from torchviz import make_dot


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(100)
    '''
    audio_model = CAVMAEFT(label_dim=cavmaeconfig['n_class'],
                           modality_specific_depth=cavmaeconfig['modality_specific_depth'])
    # for k, v in audio_model.state_dict().items():
    #      print(k)
    # mdl_weight = torch.load(cavmaeconfig['pretrain_path'])
    # if not isinstance(audio_model, torch.nn.DataParallel):
        # audio_model = torch.nn.DataParallel(audio_model)
    # miss, unexpected = audio_model.load_state_dict(mdl_weight, strict=False)
    # print('now load cav-mae pretrained weights from ', cavmaeconfig['pretrain_path'])
    # print(miss, unexpected)

    

    summary(audio_model, input_size=[(1024, 128), (3, 224, 224)], device='cpu')
    # x1 = torch.randn((2, 1024, 128)).requires_grad_(True)
    # x2 = torch.randn((2, 4, 3, 224, 224)).requires_grad_(True)
    # y = audio_model(x1, x2)
    # vis = make_dot(y, params=dict(list(audio_model.named_parameters()) + [('x1', x1), ('x2', x2)]))
    # vis.format = 'png'
    # vis.directory = '../figures/'
    # vis.view()
    '''
    train_loader, val_loader = get_dataloader(None, 15, norm=False)
    model = SyncTransformer()
    summary(model, input_size=[(4, 15*3, 96, 96), (1, 80, 1103)], device='cpu')
    print('True Total Parameters:', count_parameters(model))
    # key: state_dict, optimizer, global_step, global_epoch
    weights = torch.load('../weights/VocaLiST_Weights/vocalist_5f_lrs2.pth')
    del weights['state_dict']['classifier.weight']
    del weights['state_dict']['classifier.bias']
    # model.load_state_dict(weights['state_dict'], strict=False)
    # print('loading pretrained weights...')

    train.train_vocalist.train_vocalist(model, train_loader, val_loader, start_epoch=0)


