from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv



from data.cifar import CIFAR10, CIFAR100


if __name__ == "__main__":
    num_classes=10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])

    train_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=True, 
                                transform=transform_train,
                                noise_type='pairflip',
                                noise_rate=0.2
                            )
    
    test_dataset = CIFAR10(root='./data/',
                                download=True,  
                                train=False, 
                                transform=transform_test,
                                noise_type='pairflip',
                                noise_rate=0.2
                            )
    
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False, num_workers=2)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=100, shuffle=True, num_workers=2)

    for batch_idx, (inputs, targets, indexes) in enumerate(trainloader):
        print(inputs.shape, targets.shape, indexes.shape) 