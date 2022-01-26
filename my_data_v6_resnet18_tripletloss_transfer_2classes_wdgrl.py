import argparse
import os
import shutil
import logging
import time
import random

import torch
from torch import nn
from torch import optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# model
from models import resnet50, resnet18, CenterLoss, LabelSmoothing, resnet18_centerloss

# dataset
from dataset import MyData, MyDataTriplet, MyData_2classes, MWD_2classes
import numpy as np

from utils import AverageMeter, accuracy, Bar

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('-d', '--dataset', default='my_data', type=str)
parser.add_argument('--train-path', default="D:\\flk\\Weather_Recognition\\data_split_2classes_v6\\train\\")
parser.add_argument('--test-path', default="D:\\flk\\Weather_Recognition\\data_split_2classes_v6\\test\\")
parser.add_argument('--mwd-path', default='D:\\flk\\MWD\\weather_classification\\')
parser.add_argument('--imagesize', default=224, type=int)

parser.add_argument('--pretrained-path', default="D:\\flk\\Weather_Recognition\\checkpoints\\mwd_resnet18_2classes\\05_10_10_36_fold_0_model_best.pth.tar")
# optimization options
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=8, type=int, metavar='N',
                help='train batchsize')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                help='test batchsize')
parser.add_argument('--mwd-batch', default=24, type=int, metavar='N',
                help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.00001, type=float,
                metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40, 45],
                help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.90, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--weight-domain', type=float, default=0.1, help="weight for domain adaptation loss")
# parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH', 
#                     help='path to latest checkpoint (default: none)')

# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints/my_data_v6_resnet18_tripletloss_transfer_480_da', type=str, metavar='PATH',
                help='path to save checkpoint (default:checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH')

# architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')

# miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# use CUDA
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    start_epoch = args.start_epoch

    # 创建 checkpoint 目录
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    # load data
    # mean = [0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5]
    # imagesize = args.imagesize
    imagesize = (480, 480)
    train_transform = transforms.Compose([          
            # transforms.RandomHorizontalFlip(p=0.5),           
            # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),            
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

    valid_transform = transforms.Compose([
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

    # train_iter = MyDataTriplet(root_path=args.train_path, transforms=train_transform, batch_size=args.train_batch, shuffle=True)
    # test_iter = MyDataTriplet(root_path=args.test_path, transforms=valid_transform, batch_size=args.test_batch, shuffle=True)
    # train_dataset = MyData_2classes(root_path=args.train_path, transform=train_transform)
    test_dataset = MyData_2classes(root_path=args.test_path, transform=valid_transform)
    # dataset = MWD(root_path=args.root_path, transform=valid_transform)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    # train_iter = torch.utils.data.DataLoader(train_dataset, args.train_batch, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, args.test_batch, shuffle=True)


    # model
    model = resnet18_centerloss(pretrained=True, num_classes=2)

    # 从MWD 迁移学习 去掉全连接层
    checkpoint = torch.load(args.pretrained_path)

    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = model.state_dict()
        # print(pretrained_state_dict.keys())
    # for key in pretrained_state_dict:
    #     if((key == 'fc.weight') | (key == 'fc.bias')):

    #         pass
    #     else:
    #         model_state_dict[key] = pretrained_state_dict[key]

    # model.load_state_dict(model_state_dict)
    model.load_state_dict(pretrained_state_dict)

    model = nn.DataParallel(model)
    if use_cuda:
        model = model.cuda()

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
    # criterion
    # label_list = ['Cloud', 'Fog', 'Rainy', 'Snow', 'Sunny', 'Thunder']
    # weights = [6297, 3214, 9349, 2711, 7734, 1470]
    # normed_weights = [1 - (x / sum(weights)) for x in weights]
    # normed_weights = torch.FloatTensor(normed_weights).cuda()
    # criterion = nn.CrossEntropyLoss(weight=normed_weights)
    
    criterion = nn.CrossEntropyLoss()
    criterion_t = nn.TripletMarginLoss(margin=1.1, p=2).cuda()

    global critic, critic_optim
    critic = nn.Sequential(
        nn.Linear(515, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=1e-4)

    # criterion = LabelSmoothing(smoothing=0.1)
    # criterion = FocalLoss(5)

    # optimizer

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # set up logging
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(args.checkpoint, 'log_info.log'),
                        filemode='a+',
                        format="%(asctime)-15s %(levelname)-8s  %(message)s")
    
    # log configuration
    logging.info('-' * 10 + 'configuration' + '*' * 10)
    for arg in vars(args):
        logging.info((arg, str(getattr(args, arg))))

    # train and val
    for epoch in range(start_epoch, args.epochs):
        # 在特定的 epoch 调整学习率
        # adjust_learning_rate(optimizer, epoch)
        adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_iter = MyDataTriplet(root_path=args.train_path, transforms=train_transform, batch_size=args.train_batch, shuffle=True)

        mwd_dataset = MWD_2classes(root_path=args.mwd_path, transform=valid_transform)

        mwd_iter = torch.utils.data.DataLoader(mwd_dataset, args.mwd_batch, shuffle=True)
        mwd_iter = iter(mwd_iter)
        # mwd_iter = loop_iterable(mwd_iter)

        train_loss, train_acc = train(train_iter, mwd_iter, model, criterion, criterion_t, optimizer, epoch, use_cuda)
        test_loss, test_acc = test(test_iter, model, criterion, epoch, use_cuda)

        # logger

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),            
        }, is_best, 0, checkpoint=args.checkpoint)

        logging.info('epoch: %d, train_acc: %.2f, test_acc: %.2f' % (epoch + 1, train_acc, test_acc))

    logging.info('best_acc: %.2f' % (best_acc))


def train(train_iter, mwd_iter, model, criterion, criterion_t, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_iter))

    for batch_idx, (inputs, targets) in enumerate(train_iter):
        try:
            mwd_inputs, mwd_targets = next(mwd_iter)
        except StopIteration:
            valid_transform = transforms.Compose([
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
            ])
            mwd_dataset = MWD_2classes(root_path=args.mwd_path, transform=valid_transform)
            mwd_iter = torch.utils.data.DataLoader(mwd_dataset, args.mwd_batch, shuffle=True)
            mwd_iter = iter(mwd_iter)
            mwd_inputs, mwd_targets = next(mwd_iter)
        # measure data loading time
        data_time.update(time.time() - end)

        # debug Issues #1
        temp_batch_size = len(inputs)

        temp_x = [torch.stack(inputs[i], dim=0) for i in range(len(inputs))]
        temp_y = [torch.stack(targets[i], dim=0) for i in range(len(targets))]
        new_x = torch.stack(temp_x, dim=0)
        new_y = torch.stack(temp_y, dim=0)

        new_x = [new_x[:, i] for i in range(3)]
        new_y = [new_y[:, i] for i in range(3)]
        sample_input = torch.cat(new_x, 0)
        sample_target = torch.cat(new_y, 0)
        # print (sample_target)
        # print (sample_target[:batch_size])
        # print (sample_target[batch_size:(batch_size * 2)])
        # print (sample_target[-batch_size:])
        target = sample_target.cuda()
        inputs = sample_input.cuda()
        targets = target.cuda()


        # compute output
        feat, per_outputs = model(inputs)

        with torch.no_grad():
            mwd_feat, mwd_outputs = model(mwd_inputs)

        # domain adaptation
        size_mwd = mwd_feat.size(0)
        size_my_data = feat.size(0)

        if size_mwd == size_my_data:
            pdist = nn.PairwiseDistance(p=2)
            domain_loss = pdist(feat, mwd_feat)
            domain_loss = domain_loss.mean()
        else:
            domain_loss = 0
        anchor = feat[:temp_batch_size]
        positive = feat[temp_batch_size:(temp_batch_size * 2)]
        negative = feat[-temp_batch_size:]


        per_loss = criterion(per_outputs, targets)
        loss_triplet = criterion_t(anchor, positive, negative)
        loss = per_loss + loss_triplet + args.weight_domain * domain_loss

        # measure accuracy and record loss
        prec = accuracy(per_outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(test_iter, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(test_iter))
    for batch_idx, (inputs, targets) in enumerate(test_iter):
    # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        feat, outputs = model(inputs)
        loss = criterion(outputs, targets)

        """
        np_inputs = inputs.numpy()
        np_att = attention.numpy()
        for item_in, item_att in zip(np_inputs, np_att):
            print(item_in.shape, item_att.shape)
        """

        # measure accuracy and record loss
        prec = accuracy(outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx+1,
                    size=len(inputs),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['cloud', 'sunny'], loc='upper right')
    dirname = args.checkpoint
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()
    
def save_checkpoint(state, is_best, f_num, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'fold_' + str(f_num) + '_' + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'fold_' + str(f_num) + '_model_best.pth.tar'))


# def adjust_learning_rate(optimizer, epoch):
#     global state
#     if epoch in args.schedule:
#         state['lr'] *= args.gamma
#         for param_group in optimizer.param_groups:
#             param_group['lr'] *= args.gamma

def adjust_learning_rate(optimizer, epoch, args):
    scale_running_lr = ((1. - float(epoch) / args.epochs) ** 0.9)
    now_lr = args.lr * scale_running_lr
    # cfg.TRAIN.running_lr_encoder = cfg.TRAIN.lr_encoder * scale_running_lr
    # cfg.TRAIN.running_lr_decoder = cfg.TRAIN.lr_decoder * scale_running_lr

    # (optimizer_encoder, optimizer_decoder) = optimizers
    for param_group in optimizer.param_groups:
        param_group['lr'] = now_lr

    # for param_group in optimizer_decoder.param_groups:
    #     param_group['lr'] = cfg.TRAIN.running_lr_decoder

def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

def loop_iterable(iterable):
    while True:
        yield from iterable

if __name__ == '__main__':
    main()