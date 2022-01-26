from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable


class MyTripletLoss(nn.Module):
    def __init__(self, margin=0, gap=0.4):
        super(MyTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.triplet_loss = nn.TripletMarginLoss(margin=margin)
        self.gap = gap
    def forward(self, inputs, targets, pred):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        with torch.no_grad():
            pred = torch.nn.Softmax(dim=1)(pred)
            val, pred_2 = pred.topk(2, 1)
            confuse_idx = torch.zeros(n, dtype=torch.bool).cuda()
            for i in range(n):
                if val[i][0] - val[i][1] <= self.gap:
                    confuse_idx[i] = True
                if pred_2[i][0] != targets[i]:
                    confuse_idx[i] = True
        

        # with torch.no_grad():
        #     pred = torch.nn.Softmax(dim=1)(pred)
        #     val, pred_2 = pred.topk(2, 1)
        #     confuse_idx = torch.zeros(n).cuda()
        #     for i in range(n):
        #         confuse_idx[i] = val[i][0] - val[i][1]
        #         if pred_2[i][0] != targets[i]:
        #             confuse_idx[i] = 1 + pred_2[i][0]

        # for i in range(n // 2):
        #     dist_ap.append(dist[rank[i]][mask[i]].max())
        #     dist_an.append(dist[rank[i]][mask[i] == 0].min())

        # _, rank = confuse_idx.topk(n, 0)
        # group = torch.zeros(n, dtype=torch.bool).cuda()
        # group[rank[n//2:]] = True

        # for i in range(n // 2):

        #     mask_certain = mask[i] & (group)
        #     mask_confuse = (~mask[i]) & (group)
        #     if not mask_certain.any() or not mask_confuse.any():
        #         continue
        #     dist_ap.append(dist[i][mask_certain].max())
        #     dist_an.append(dist[i][mask_confuse].min())
        #     # dist_ap.append(dist[i][mask[i]].max())
        #     # dist_an.append(dist[i][mask[i] == 0].min())


        for i in range(n):
            if confuse_idx[i] == True:

                # mask_certain = mask[i] & (~confuse_idx)
                # mask_confuse = (~mask[i]) & (~confuse_idx)
                # if not mask_certain.any() or not mask_confuse.any():
                #     continue
                # dist_ap.append(dist[i][mask_certain].max())
                # dist_an.append(dist[i][mask_confuse].min())
                dist_ap.append(dist[i][mask[i]].max())
                dist_an.append(dist[i][mask[i] == 0].min())
        # dist_ap = torch.cat(dist_ap)
        # dist_an = torch.cat(dist_an)
        if len(dist_an) == 0 or len(dist_ap) == 0:
            return 0
        dist_ap = torch.FloatTensor(dist_ap)
        dist_an = torch.FloatTensor(dist_an)
        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
        # return loss, prec
        return loss


def my_hard_triplet_loss(labels, embeddings, margin, pred, gap=0.3, squared=False):
    """
    - compute distance matrix
    - for each anchor a0, find the (a0,p0) pair with greatest distance s.t. a0 and p0 have the same label
    - for each anchor a0, find the (a0,n0) pair with smallest distance s.t. a0 and n0 have different label
    - compute triplet loss for each triplet (a0, p0, n0), average them
    """
    _, pred_2 = outputs.topk(2, 1)
    b_z = labels.size(0)
    confuse_idx = torch.zeros(bz)
    for i in range(b_z):
        if pred_2[i][0] - pred_2[i][1] <= gap:
            idx[i] = 1
    
    certern_idx = ~confuse_idx
    confuse_group = embeddings[confuse_idx]
    certern_group = embeddings[certern_idx]

    mask = targets.expand(n, n).eq(targets.expand(n, n).t())

    distances = pairwise_distances(embeddings, squared=squared)

    mask_positive = get_valid_positive_mask(labels)
    hardest_positive_dist = (distances * mask_positive.float()).max(dim=1)[0]

    mask_negative = get_valid_negative_mask(labels)
    max_negative_dist = distances.max(dim=1,keepdim=True)[0]
    distances = distances + max_negative_dist * (~mask_negative).float()
    hardest_negative_dist = distances.min(dim=1)[0]

    triplet_loss = (hardest_positive_dist - hardest_negative_dist + margin).clamp(min=0)
    triplet_loss = triplet_loss.mean()

    return triplet_loss