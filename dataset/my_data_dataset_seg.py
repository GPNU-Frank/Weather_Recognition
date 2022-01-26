from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch
from sklearn.model_selection import train_test_split
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyDataSeg(Dataset):
    def __init__(self, root_path, transform=None, is_train=True):
        self.root_path = root_path
        self.num_classes = 6
        self.transform = transform
        # label_dict = {'Cloud': 0, 'Fog': 1, 'Rainy': 2, 'Snow': 3, 'Sunny': 4, 'Thunderstorm': 5}
        label_dict = {'Cloud': 0, 'Fog': 1, 'Rainy': 2, 'Snow': 3, 'Sunny': 4}

        if not os.path.exists(root_path):
            print(root_path)
            raise FileNotFoundError
        self.img_list = []
        with os.scandir(root_path) as root:
            for emotion in root:
                label = label_dict[emotion.name]
                with os.scandir(emotion.path) as fold:
                    for img in fold:
                        img_seg = img.path.replace("data_split", "data_split_road_segment")
                        if img_seg == img:
                            raise Exception("road image path wrong")
                        self.img_list.append((img.path, img_seg, label))
    
    def __getitem__(self, index):
        img_path, img_seg_path, label = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        img_seg = Image.open(img_seg_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
            img_seg = self.transform(img_seg)
            
        return img, img_seg, label

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    
    train_path = "G:\\vscode_workspace\\Weather_Recognition\\data_split_v4\\train\\"
    test_path = "G:\\vscode_workspace\\Weather_Recognition\\data_split_v4\\test\\"
    # root_path = "G:\\dataset\\MWD\\weather_classification\\"
    imagesize = 224
    # transform = transforms.Compose(
    #     [transforms.Resize((imagesize, imagesize)), transforms.ToTensor()])

    transform = transforms.Compose(
        [transforms.Resize((imagesize, imagesize))])

    train_dataset = MyDataSeg(root_path=train_path, transform=transform)
    test_dataset = MyDataSeg(root_path=test_path, transform=transform)
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])
    # print(dataset[0: 10])
    # x_train, x_test, y_train, y_test = train_test_split(dataset[:][0], dataset[:][1], test_size=0.3, stratify=dataset[:][1])
    # print(len(dataset))
    print(len(train_dataset))
    print(len(test_dataset))
    # print(len(train_dataset), len(test_dataset))
    # print(dataset[0][0].shape)