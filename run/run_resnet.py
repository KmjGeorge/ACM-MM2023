from model.cnntest import train_resnet50
import torch
from configs.nsconfig import cavmaeconfig, trainconfig
from dataset.dataloader import get_dataloader, get_testloader
import numpy as np

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
    resnet50 = get_resnet50()
    # weights = torch.load('../weights/resnet50-1e-5 0.95_epoch30.pt')
    # new_weights = {}
    # for k, v in weights.items():
    #     new_k = k.replace('module.', '')
    #     new_weights[new_k] = v
    # resnet50.load_state_dict(new_weights)
    train_loader, val_loader = get_dataloader('mean')
    summary(resnet50, input_size=(3, 224, 224), device='cpu')
    train_resnet50(resnet50, train_loader, val_loader, start_epoch=0)
