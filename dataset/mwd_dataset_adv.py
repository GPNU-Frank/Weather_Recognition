from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np


class MWD_Adv(Dataset):
    def __init__(self, root_path, transform=None, is_train=True):
        self.root_path = root_path
        self.num_classes = 6
        self.transform = transform
        label_dict = {'cloudy': 0, 'haze': 1, 'rainy': 2, 'snow': 3, 'sunny': 4, 'thunder': 5}

        if not os.path.exists(root_path):
            print(root_path)
            raise FileNotFoundError
        self.img_list = []
        with os.scandir(root_path) as root:
            for emotion in root:
                label = label_dict[emotion.name]
                with os.scandir(emotion.path) as fold:
                    for img in fold:
                        self.img_list.append((img.path, label))
    
    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = Image.open(img_path).convert('RGB')

        img_contrast = self.get_contrast_intensity(img)
        img_hist = self.get_saturation_hist(img)

        # 最大最小值归一化
        img_contrast = (img_contrast - np.mean(img_contrast)) / np.std(img_contrast)

        img_hist = (img_hist - np.mean(img_hist)) / np.std(img_hist)
        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_contrast, img_hist

    def __len__(self):
        return len(self.img_list)

    def get_contrast_intensity(self, img):
        img = np.array(img)
        hight, width, _ = img.shape

        row, col = 4, 4
        img_patch = [[None] * col for _ in range(row)]
        img_contrast = [[None] * col for _ in range(row)]

        for i in range(row):
            for j in range(col):
                img_patch[i][j] = img[hight // row * i: hight // row * (i + 1), width // col * j: width // col * (j + 1), :]
                gray_img = cv2.cvtColor(img_patch[i][j], cv2.COLOR_BGR2GRAY)
                img_contrast[i][j] = gray_img.std()
        
        return np.array(img_contrast).flatten().astype(np.float32)

    def get_saturation_hist(self, img):
        img = np.array(img)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([img_hsv], [1], None, [256], [0.0, 255.0])
        hist = np.array(hist)
        return np.squeeze(hist, axis=-1)

if __name__ == '__main__':
    root_path = "G:\\dataset\\MWD\\weather_classification\\"
    imagesize = 224
    transform = transforms.Compose(
        [transforms.Resize((imagesize, imagesize)), transforms.ToTensor()])

    dataset = MWD_Adv(root_path=root_path, transform=transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

    print(len(dataset))
    print(len(train_dataset), len(test_dataset))
    print(dataset[0][0].shape)