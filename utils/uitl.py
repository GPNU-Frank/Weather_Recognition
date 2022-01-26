import numpy as np
import torch
import pickle
import torchvision
import matplotlib.pyplot as plt

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def data_loader_with_LOSO(index, onset_imgs, apex_imgs, optical_flows, labels):
    train_x, train_y, test_x, test_y = [], [], [], []

    test_x.append((onset_imgs[index], apex_imgs[index], optical_flows[index]))
    test_y.append(labels[index])

    for i in range(len(labels)):
        if i != index:
            train_x.append((onset_imgs[i], apex_imgs[i], optical_flows[i]))
            train_y.append(labels[i])

    return np.vstack(train_x), np.vstack(train_y), np.vstack(test_x), np.vstack(test_y)


def load_parameter(_structure, _parameterDir):

    # resnet18
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']

    print(pretrained_state_dict.keys())
    model_state_dict = _structure.state_dict()
    print(model_state_dict.keys())
    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    # model = torch.nn.DataParallel(_structure).cuda()

    return _structure


def load_state_dict(model, fname):

    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    print(weights.keys())
    for name, param in weights.items():
        if name in own_state:
            if name in ['fc.weight', 'fc.bias']:
                continue
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError('While copying the parameter named {}, whose dimensions in the model are {} and whose '\
                                   'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def imshow_batch(images, labels):
    """Displays two grids of images. The top grid displays ``images``
    and the bottom grid ``labels``
    Keyword arguments:
    - images (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    - labels (``Tensor``): a 4D mini-batch tensor of shape
    (B, C, H, W)
    """

    # Make a grid with the images and labels and convert it to numpy
    images = torchvision.utils.make_grid(images).numpy()
    labels = torchvision.utils.make_grid(labels).numpy()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 7))
    ax1.imshow(np.transpose(images, (1, 2, 0)))
    ax2.imshow(np.transpose(labels, (1, 2, 0)))

    plt.show()


def batch_transform(batch, transform):
    """Applies a transform to a batch of samples.
    Keyword arguments:
    - batch (): a batch os samples
    - transform (callable): A function/transform to apply to ``batch``
    """

    # Convert the single channel label to RGB in tensor form
    # 1. torch.unbind removes the 0-dimension of "labels" and returns a tuple of
    # all slices along that dimension
    # 2. the transform is applied to each slice
    transf_slices = [transform(tensor) for tensor in torch.unbind(batch)]

    return torch.stack(transf_slices)