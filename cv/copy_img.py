import shutil
import os

if __name__ == "__main__":

    root_path = "G:\\vscode_workspace\\Weather_Recognition\\data_v6_new\\train2\\"
    copy_path = "G:\\vscode_workspace\\Weather_Recognition\\data_v6_new\\train\\"
    # label_dict = {'Cloud': 0, 'Sunny': 2}
    label_dict = {'Cloud': 0, 'Fog': 1, 'Rainy': 2, 'Snow': 3, 'Sunny': 4}
    with os.scandir(root_path) as root:
            for emotion in root:
                if emotion.name in label_dict:
                    label = label_dict[emotion.name]
                    with os.scandir(emotion.path) as fold:
                        for camera in fold:
                            with os.scandir(camera.path) as site:
                                for img in site:
                                    target_path = copy_path + emotion.name + os.sep + img.name
                                    shutil.copy(img.path, target_path)
                                    # self.img_list.append((img.path, label))