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
from models import resnet50, resnet18, FocalLoss
from models.label_smoothing import LabelSmoothing

# dataset
from dataset import MyData
import numpy as np

from utils import AverageMeter, accuracy, Bar

# scheduler
from scheduler import StepLRScheduler

parser = argparse.ArgumentParser()

# datasets
parser.add_argument('-d', '--dataset', default='my_data', type=str)
parser.add_argument('--train-path', default="G:\\vscode_workspace\\Weather_Recognition\\data3\\train\\")
parser.add_argument('--test-path', default="G:\\vscode_workspace\\Weather_Recognition\\data3\\test\\")
parser.add_argument('--imagesize', default=224, type=int)

parser.add_argument('--pretrained-path', default="G:\\vscode_workspace\\Weather_Recognition\\checkpoints\\from_others\\mwd_resnet_384_model_best.pth.tar")

# optimization options
# parser.add_argument('--epochs', default=20, type=int, metavar='N',
#                 help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                 help='manual epoch number (useful on restarts)')
parser.add_argument('--train_batch', default=8, type=int, metavar='N',
                help='train batchsize')
parser.add_argument('--test-batch', default=8, type=int, metavar='N',
                help='test batchsize')
# parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
#                 metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10, 15, 20, 25, 30, 35, 40, 45],
                help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.90, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.8, type=float, metavar='M',
                help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                metavar='W', help='weight decay (default: 1e-4)')

# parser.add_argument('--pretrained', default='pretrainedmodels/vgg_msceleb_resnet50_ft_weight.pkl', type=str, metavar='PATH', 
#                     help='path to latest checkpoint (default: none)')

# checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoints/data3_resnet18_transfer_480_lr_sche_norm', type=str, metavar='PATH',
                help='path to save checkpoint (default:checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH')

# architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18')

# miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')



# Learning rate schedule parameters
parser.add_argument('--sched', default='step', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit')
parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10')
parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')


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
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    # imagesize = args.imagesize
    imagesize = (480, 480)
    train_transform = transforms.Compose([          
            transforms.RandomHorizontalFlip(p=0.5),           
            # transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.25, hue=0.05),            
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(imagesize),
            # cutout(64, 1, False),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    valid_transform = transforms.Compose([
            # transforms.Resize((args.imagesize, args.imagesize)),
            transforms.Resize(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = MyData(root_path=args.train_path, transform=train_transform)
    test_dataset = MyData(root_path=args.test_path, transform=valid_transform)
    # dataset = MWD(root_path=args.root_path, transform=valid_transform)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    train_iter = torch.utils.data.DataLoader(train_dataset, args.train_batch, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, args.test_batch, shuffle=True)
    # model
    model = resnet18(pretrained=True)

    # 从MWD 迁移学习 去掉全连接层
    checkpoint = torch.load(args.pretrained_path)

    pretrained_state_dict = checkpoint['state_dict']
    # model_state_dict = model.state_dict()
        # print(pretrained_state_dict.keys())
    # for key in pretrained_state_dict:
    #     if((key == 'fc.weight') | (key == 'fc.bias')):

    #         pass
    #     else:
    #         model_state_dict[key] = pretrained_state_dict[key]

    model.load_state_dict(pretrained_state_dict)

    # model = nn.DataParallel(model)
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
    # criterion = LabelSmoothing(smoothing=0.1)
    # criterion = FocalLoss(5)

    # optimizer

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # lr_scheduler
    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * args.epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * args.epochs
    else:
        noise_range = None

    lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_epochs,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epochs,
            noise_range_t=noise_range,
            noise_pct=getattr(args, 'lr_noise_pct', 0.67),
            noise_std=getattr(args, 'lr_noise_std', 1.),
            noise_seed=getattr(args, 'seed', 42),
        )

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
        # adjust_learning_rate(optimizer, epoch, args)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        train_loss, train_acc = train(train_iter, model, criterion, optimizer, epoch, use_cuda, lr_scheduler=lr_scheduler)
        test_loss, test_acc = test(test_iter, model, criterion, epoch, use_cuda)

        # step scheduler
        lr_scheduler.step(epoch)
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


def train(train_iter, model, criterion, optimizer, epoch, use_cuda, lr_scheduler=None):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_iter))

    num_updates = epoch * len(train_iter)
    for batch_idx, (inputs, targets) in enumerate(train_iter):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        per_outputs = model(inputs)

        per_loss = criterion(per_outputs, targets)

        loss = per_loss

        # measure accuracy and record loss
        prec = accuracy(per_outputs.data, targets.data, topk=(1,))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec[0].item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        num_updates += 1
        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates)

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
        outputs = model(inputs)
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


def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

if __name__ == '__main__':
    main()