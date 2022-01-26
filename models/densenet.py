import torch
from torch import nn
from torchvision import models


def densenet_121(pretrained=False):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, 5) 
    return model


def densenet_121_2classes(pretrained=False):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, 2) 
    return model

if __name__ == "__main__":
    model = densenet_121(pretrained=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # print(model)