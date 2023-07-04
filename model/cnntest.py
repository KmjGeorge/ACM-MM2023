import torchvision
import torch

from configs.nsconfig import *

def get_resnet50(pretrained=True):
    net = torchvision.models.resnet50(pretrained=False)
    if pretrained:
        net.load_state_dict(torch.load('../weights/resnet50-0676ba61.pth'))
    del net.fc
    net.add_module('fc', nn.Linear(2048, 4))
    return net
