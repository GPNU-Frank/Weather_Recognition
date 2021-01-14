import torch
from torch import nn
from torchvision import models


def densenet_121(pretrained=False):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, 6) 
    return model


if __name__ == "__main__":
    model = densenet_121(pretrained=True)
    print(model)