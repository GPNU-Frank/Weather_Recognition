from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import os
import torch

class MWD(Dataset):
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
                # if emotion.name == 'thunder':
                #     continue
                label = label_dict[emotion.name]
                with os.scandir(emotion.path) as fold:
                    for img in fold:
                        self.img_list.append((img.path, label))
    
    def __getitem__(self, index):
        img_path, label = self.img_list[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.img_list)

if __name__ == '__main__':
    root_path = "G:\\dataset\\MWD\\weather_classification\\"
    imagesize = 224
    transform = transforms.Compose(
        [transforms.Resize((imagesize, imagesize)), transforms.ToTensor()])

    dataset = MWD(root_path=root_path, transform=transform)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [50000, 10000])

    print(len(dataset))
    print(len(train_dataset), len(test_dataset))
    print(dataset[0][0].shape)