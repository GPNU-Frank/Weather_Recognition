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
from models import resnet50, resnet18, resnet18_seg, FocalLoss, resnet18_seg_snow

# dataset
from dataset import MyData, MyDataSeg, MyDataSegSnow
import numpy as np

from utils import AverageMeter, accuracy, Bar

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('-d', '--dataset', default='my_data_seg', type=str)
parser.add_argument('--train-path', default="G:\\vscode_workspace\\Weather_Recognition\\data_split_v4\\train\\")
parser.add_argument('--test-path', default="G:\\vscode_workspace\\Weather_Recognition\\data_split_v4\\test\\")
parser.add_argument('--imagesize', default=224, type=int)

# optimization options
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=4, type=int, metavar='N',
                help='train batchsize')
parser.add_argument('--test-batch', default=4, type=int, metavar='N',
                help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[1, 3, 5, 7, 9],
                help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.80, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                metavar='W', help='weight decay (default: 1e-4)')

# parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH', 
#                     help='path to latest checkpoint (default: none)')

# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints/my_data_v4_resnet18_seg_snow', type=str, metavar='PATH',
                help='path to save checkpoint (default:checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH')

# parser.add_argument('--resume', default='checkpoints/my_data_v4_resnet18_seg_4classes/fold_0_checkpoint.pth.tar', type=str, metavar='PATH')

# architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_seg_snow')

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
    image_size = (576, 720)

    # 使用图片原尺寸
    train_transform = transforms.Compose([          
            # transforms.RandomHorizontalFlip(p=0.5),           
            # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),            
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

    valid_transform = transforms.Compose([
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean, std)
        ])

    train_dataset = MyDataSegSnow(root_path=args.train_path, transform=train_transform)
    test_dataset = MyDataSegSnow(root_path=args.test_path, transform=valid_transform)
    # dataset = MWD(root_path=args.root_path, transform=valid_transform)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    train_iter = torch.utils.data.DataLoader(train_dataset, args.train_batch, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, args.test_batch, shuffle=False)
    # model
    model = resnet18_seg_snow(pretrained=True)
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
    criterion_snow = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(5)

    # optimizer
    # optimizer
    seg_layer3_params = list(map(id, model.seg_layer3.parameters()))
    fc_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in seg_layer3_params + fc_params,
                            model.parameters())

    optimizer = optim.SGD([{'params': base_params},
                            {'params': model.seg_layer3.parameters(), 'lr': 0.005},
                            {'params': model.fc.parameters(), 'lr': 0.005}
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
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
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(train_iter, model, criterion, optimizer, epoch, use_cuda, criterion_snow=criterion_snow)
        # 清理显存
        torch.cuda.empty_cache()
        test_loss, test_acc = test(test_iter, model, criterion, epoch, use_cuda, criterion_snow=criterion_snow)
        # 清理显存
        torch.cuda.empty_cache()

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


def train(train_iter, model, criterion, optimizer, epoch, use_cuda, criterion_snow=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_iter))

    for batch_idx, (inputs, inputs_seg, targets, targets_snow) in enumerate(train_iter):
        # measure data loading time
        data_time.update(time.time() - end)

        targets_snow = targets_snow.float()
        if use_cuda:
            inputs, inputs_seg, targets, targets_snow = inputs.cuda(), inputs_seg.cuda(), targets.cuda(), targets_snow.cuda()

        # compute output
        per_outputs, snow_outputs = model(inputs, inputs_seg)

        per_loss = criterion(per_outputs, targets)

        snow_loss = criterion_snow(snow_outputs, targets_snow)

        loss = per_loss + snow_loss

        # measure accuracy and record loss
        prec = accuracy(per_outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 清理显存
        # torch.cuda.empty_cache()
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


def test(test_iter, model, criterion, epoch, use_cuda, criterion_snow=None):
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
    for batch_idx, (inputs, inputs_seg, targets, targets_snow) in enumerate(test_iter):
    # measure data loading time
        data_time.update(time.time() - end)


        targets_snow = targets_snow.float()
        if use_cuda:
            inputs, inputs_seg, targets, targets_snow = inputs.cuda(), inputs_seg.cuda(), targets.cuda(), targets_snow.cuda()
        # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs, outputs_snow = model(inputs, inputs_seg)
        loss = criterion(outputs, targets)

        loss += criterion_snow(outputs_snow, targets_snow)

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


def save_checkpoint(state, is_best, f_num, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, 'fold_' + str(f_num) + '_' + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'fold_' + str(f_num) + '_model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.gamma


if __name__ == '__main__':
    main()