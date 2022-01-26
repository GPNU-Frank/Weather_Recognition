import torch
from torch import nn
from torchvision import models
from torchvision.models import vgg


def vgg_13(pretrained=False):
    model = models.vgg13(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 5) 
    return model

def vgg_16(pretrained=False):
    model = models.vgg16(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 5) 
    return model


def densenet_121_2classes(pretrained=False):
    model = models.densenet121(pretrained=pretrained)
    model.classifier = nn.Linear(1024, 2) 
    return model

if __name__ == "__main__":
    model = vgg_13(pretrained=True)
    # model = vgg_16(pretrained=True)
    
    inputs = torch.randn(1, 3, 224, 224)
    
    from thop import profile
    flops, params = profile(model, (inputs,))
    print('flops: ', flops/1000000.0, 'params: ', params/1000000.0)

    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # print(model)