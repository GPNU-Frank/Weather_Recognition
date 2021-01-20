
import torch
import numpy as np
import cv2
from PIL import Image

import sys
sys.path.append('..')
from models import resnet50, densenet_121, resnet50_adv
from utils import *

# 加载真实路面数据集
from dataset import MWD, MWD_Adv
from dataset import MyData
from torchvision import transforms
import matplotlib.pyplot as plt
import itertools
# checkpoint_path = '../checkpoints/mwd_resnet50/fold_0_model_best.pth.tar'
# checkpoint_path = '../checkpoints/my_data_resnet50/fold_0_model_best.pth.tar'
# model = resnet50()
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['state_dict'])

# checkpoint_path = '../checkpoints/my_data_densenet121/fold_0_model_best.pth.tar'
# model = densenet_121()
# checkpoint = torch.load(checkpoint_path)
# model.load_state_dict(checkpoint['state_dict'])

checkpoint_path = '../checkpoints/my_data_resnet50_cs/fold_0_model_best.pth.tar'
model = resnet50_adv()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['state_dict'])


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    # transforms.Normalize(mean, std)
])

# dataset = MWD(root_path="G:\\weather_recognition\\groud_truth", transform=transform)
dataset = MWD_Adv(root_path="G:\\weather_recognition\\groud_truth", transform=transform)

# train_dataset = MyData(root_path=args.train_path, transform=train_transform)
# dataset = MyData(root_path="G:\\vscode_workspace\\Weather_Recognition\\data_split_v2\\test\\", transform=transform)


print(len(dataset))
print(dataset[0][0].shape)


# img, true label, pred label 可视化
label_dict = {'cloudy': 0, 'haze': 1, 'rainy': 2, 'snow': 3, 'sunny': 4, 'thunder': 5}
label_list = ['cloudy', 'haze', 'rainy', 'snow', 'sunny', 'thunder']
def show_model_performance(images, labels, outputs, batch_idx=0):
    # print(images.shape, labels.shape, outputs.shape)
    _, figs = plt.subplots(1, 32, figsize=(96, 96))
    # print(len(figs))
    for f, img, lbl, pred in zip(figs, images, labels, outputs):
        img = np.transpose(img, (1, 2, 0))
        f.imshow(img)
        # print(lbl, pred)
        f.set_title(label_list[lbl] + ';' + label_list[pred[0]])
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    
    fig = plt.gcf()
    fig.savefig('figs/show_performance_' + str(batch_idx) + '.png', format='png', transparent=True)

    plt.show()


# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_cm(cm, preds, labels):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        # if t == 2 and p == 1:
        #     print('miss!')
        # cm[p, t] += 1
        cm[t, p] += 1
    return cm


cnt = 0
def save_misclassify(inputs, targets, preds):
    global cnt
    save_dir = "misclassify_01_14_resnet_adv_before//"
    label_list = ['Cloud', 'Fog', 'Rainy', 'Snow', 'Sunny', 'Thunder']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for img, t, p in zip(inputs, targets, preds):
        p = p[0]
        # print(img.shape, t, p)
        # break
        if t != p:
            img = np.transpose(img, (1, 2, 0))
            img = img * 255
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # print(img.shape)
            img_path = save_dir + '{0}_{1}_{2}.jpg'.format(cnt, label_list[t], label_list[p]) 
            cv2.imwrite(img_path, img)
            cnt = cnt + 1



# # 'cloudy': 0, 'haze': 1, 'rainy': 2, 'snow': 3, 'sunny': 4, 'thunder': 5
# model = model.cuda()
# model.eval()
# top1 = AverageMeter()
# data_iter = torch.utils.data.DataLoader(dataset, 32, shuffle=False)

# # 混淆矩阵
# num_classes = 6
# label_list = ['Cloud', 'Fog', 'Rainy', 'Snow', 'Sunny', 'Thunder']
# confuse_matrix = np.zeros([num_classes, num_classes])
# for batch_idx, (inputs, targets) in enumerate(data_iter):
#     inputs, targets = inputs.cuda(), targets.cuda()

#     outputs = model(inputs)

#     # print(outputs)
#     _, pred = outputs.topk(1, 1)
#     # print(pred)
#     # print(targets)
#     confuse_matrix = compute_cm(confuse_matrix, outputs, targets)
#     prec = accuracy(outputs.data, targets.data, topk=(1,))
#     top1.update(prec[0].item(), inputs.size(0))
#     # save_misclassify(inputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), pred.cpu().detach().numpy())
#     # break
# print(top1.avg)
# print(confuse_matrix)
# plot_confusion_matrix(confuse_matrix, label_list, normalize=True)
#     # show_model_performance(inputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), pred.cpu().detach().numpy(), batch_idx)
#     # break



# 'cloudy': 0, 'haze': 1, 'rainy': 2, 'snow': 3, 'sunny': 4, 'thunder': 5
model = model.cuda()
model.eval()
top1 = AverageMeter()
data_iter = torch.utils.data.DataLoader(dataset, 16, shuffle=False)

# 混淆矩阵
num_classes = 6
label_list = ['Cloud', 'Fog', 'Rainy', 'Snow', 'Sunny', 'Thunder']
confuse_matrix = np.zeros([num_classes, num_classes])
for batch_idx, (inputs, targets, constrast, hist) in enumerate(data_iter):
    inputs, targets, constrast, hist = inputs.cuda(), targets.cuda(), constrast.cuda(), hist.cuda()

    outputs = model(inputs, constrast, hist)

    # print(outputs)
    _, pred = outputs.topk(1, 1)
    # print(pred)
    # print(targets)
    confuse_matrix = compute_cm(confuse_matrix, outputs, targets)
    prec = accuracy(outputs.data, targets.data, topk=(1,))
    top1.update(prec[0].item(), inputs.size(0))
    # save_misclassify(inputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), pred.cpu().detach().numpy())
    # break
print(top1.avg)
print(confuse_matrix)
plot_confusion_matrix(confuse_matrix, label_list, normalize=True)
    # show_model_performance(inputs.cpu().detach().numpy(), targets.cpu().detach().numpy(), pred.cpu().detach().numpy(), batch_idx)
    # break




