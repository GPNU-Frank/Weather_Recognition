from torch import nn
import torch
import torch.nn.functional as F


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        print(weight)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        print(weight)
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class MySmoothLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, weights, classes=5):
        inputs = F.log_softmax(inputs, dim=-1)
        # print(input)
        max_prob, _ = inputs.max(dim=1, keepdim=True)
        # print(max_prob)
        max_weight, _ = weights.max(dim=1, keepdim=True)
        smooth_mask = max_weight < 0.8
        # print(smooth_mask)
        confidence = (1 - 0.1) * max_weight * smooth_mask
        # print(confidence)
        smooth = confidence / (classes - 1.0)
        # print(smooth)
        smooth = smooth.expand(inputs.size(0), classes)
        # print(targets.data.unsqueeze(1))
        # print(smooth.dtype)
        # print(smooth)
        # smooth = torch.zeros(2, 5)
        # smooth = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        # [0.0600, 0.0600, 0.0600, 0.0600, 0.0600]])
        # smooth = torch.tensor(smooth)
        smooth = smooth.clone()
        # print(smooth)
        smooth.scatter_(1, targets.unsqueeze(1), 1-confidence)
        # print(smooth)
        loss = (-smooth * inputs).sum(dim=-1).mean()
        return loss
        # return smooth


if __name__ == "__main__":
    inputs = [[0.1, 0.1, 0.1, 0.0, 0.7],
                [0.4, 0.1, 0.0, 0.0, 0.6]]
    targets = [4, 4]
    weights = [[0.1, 0.1, 0.1, 0.0, 0.7],
                [0.4, 0.1, 0.0, 0.0, 0.6]]
    inputs = torch.tensor(inputs)
    targets = torch.LongTensor(targets)
    weights = torch.tensor(weights)

    criterion = MySmoothLoss()
    # criterion1 = LabelSmoothLoss(smoothing=0.1)
    # criterion1(inputs, targets)
    result = criterion(inputs, targets, weights)
    print(result)