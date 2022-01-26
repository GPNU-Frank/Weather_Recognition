import torch
from torch import nn
from torchvision import models
from torchvision.models import vgg


def vgg_13(pretrained=False):
    model = models.vgg13(pretrained=pretrained)
    model.classifier[6] = nn.Linear(4096, 5) 
    return model

def google_net(pretrained=False):
    model = models.googlenet(pretrained=pretrained)
    model.fc = nn.Linear(1024, 5)
    return model
    



if __name__ == "__main__":
    model = google_net(pretrained=True)

    inputs = torch.randn(1, 3, 224, 224)
    
    from thop import profile
    flops, params = profile(model, (inputs,))
    print('flops: ', flops/1000000.0, 'params: ', params/1000000.0)
    # print(model)
    # print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    # print(model)