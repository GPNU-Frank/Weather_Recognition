import cv2
import PIL
import numpy as np
import os


def procecss_hull(img_path, img_seg_path, save_path):
    img = cv2.imread(img_seg_path)
    ori_img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    ori_img = cv2.resize(ori_img, (704, 576))
    # print(ori_img.shape)
    area = []
    
    if len(contours) == 0:
        print("wrong contour ", img_path)
        return
    # 找到最大的轮廓
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))

    # 填充最大的轮廓
    cnt = contours[max_idx]

    hull = cv2.convexHull(cnt)
    # cv2.polylines(img, [hull], True, (0, 0, 255), 2)
    cv2.polylines(ori_img, [hull], True, (0, 0, 255), 2)

    cv2.imwrite(save_path, ori_img)


if __name__ == "__main__":
    root_path = "G:\\vscode_workspace\\Weather_Recognition\\data_split_v6\\train\\"

    if not os.path.exists(root_path):
        print(root_path)
        raise FileNotFoundError

    with os.scandir(root_path) as root:
        for emotion in root:

            with os.scandir(emotion.path) as fold:

                for img in fold:

                    img_seg_hull_path = img.path.replace("data_split", "data_split_segment_hull")

                    img_seg_path = img.path.replace("data_split", "data_split_road_segment")

                    # hull_fold = "\\".join(img_seg_hull_path.split('\\')[:-1])
                    # if not os.path.exists(hull_fold):
                    #     os.makedirs(hull_fold)

                    procecss_hull(img.path, img_seg_path, img_seg_hull_path)
                    # break
