import numpy as np
import os
import shutil
from collections import defaultdict
import random


if __name__ == "__main__":
    # 标签列表
    # classes_list = ["Cloud", "Fog", "Rainy", "Snow", "Sunny", "Thunderstorm"]
    classes_list = ["Cloud", "Fog", "Rainy", "Snow", "Sunny"]
    root_path = 'data/'


    with os.scandir(root_path) as root:
        for label in root:
            if label.name in classes_list:
                # 读每一类下面的图片
                with os.scandir(label.path) as fold:
                    map_dict = defaultdict(list)
                    for img in fold:
                        img_name = img.name
                        # print(img_name)
                        # break
                        img_group_name = img_name.split('_')[0]
                        
                        # 根据 pid 聚合同一个视频流的视频 （适用 Cloud Fog Rainy Snow Sunny）
                        map_dict[img_group_name].append((img.path, img.name))
                        # print(img_group_name)
                        # break

                    # 按 7:3 的比例打乱并划分训练集测试集
                    group_list = list(map_dict)

                    length = len(group_list)
                    offset = int(length * 0.7)
                    if length == 0 or offset < 1:
                        print("异常 " + label.name)
                    random.shuffle(group_list)
                    train_list = group_list[:offset]
                    test_list = group_list[offset:]

                    # 分别保存
                    train_path = root_path + os.sep + 'train' + os.sep + label.name
                    test_path = root_path + os.sep + 'test' + os.sep + label.name

                    if not os.path.exists(train_path):
                        os.makedirs(train_path)
                    
                    if not os.path.exists(test_path):
                        os.makedirs(test_path)

                    # 根据原图片路径直接复制图片到目标路径中
                    for group in train_list:
                        for img_path, img_name in group:
                            copy_path = train_path + os.sep + img_name
                            shutil.copyfile(img_path, copy_path)

                    for group in test_list:
                        for img_path, img_name in group:
                            copy_path = test_path + os.sep + img_name
                            shutil.copyfile(img_path, copy_path)
                    
                    


