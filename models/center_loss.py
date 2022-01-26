import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

import random
import numpy as np
if __name__ == "__main__":
    feat = torch.rand(50, 512)
    label = torch

    label = []
    for i in range(50):
        num = random.randint(0, 1)
        label.append(num)
    label = np.array(label)
    label = torch.tensor(label, dtype=int)

    select = torch.where(label == 1)
    select = select[0]

    select0 = torch.where(label == 0)
    select0 = select0[0]
    # print(select)
    print(select.shape)
    print(label)
    print(feat[select].shape)

    size_1 = select.size()[0]
    center_1 = feat[select].sum(dim=0) / size_1

    size_0 = select0.size()[0]
    center_0 = feat[select0].sum(dim=0) / size_0
    print(center_1.shape)
    print(center_0.shape)

    dist = (center_1 - center_0) ** 2
    res = torch.sqrt(dist.sum())
    print(res)

    # dist = torch.abs()