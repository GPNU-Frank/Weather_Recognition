import cv2
import numpy as np
import matplotlib.pyplot as plt
from color_transfer import color_transfer
import os
import random

def generate_thunder(src_path, thunder_path):

    # 读源图片，canny边缘检测，膨胀操作
    img_src = cv2.imread(src_path)

    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_src_gray = cv2.GaussianBlur(img_src_gray, (5, 5), 0)
    img_src_canny = cv2.Canny(img_src_gray, 100, 140)

    src_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    src_dilated = cv2.dilate(img_src_canny, src_kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(src_dilated, connectivity=8, ltype=cv2.CV_32S)
    hight, width = src_dilated.shape
    print(num_labels)

    # 筛选连通区域
    best_x, best_y, best_w, best_h, best_area = 0, 0, 0, 0, 0
    sky_line_num = -1
    for idx, (x, y, w, h, area) in enumerate(stats):
        if w >= width / 3 and (x != 0 or y != 0):
            # print(x, y, w, h, area)
            if y + h < best_y + best_h or best_area == 0:
                best_x, best_y, best_w, best_h, best_area = x, y, w, h, area
                # flag = 0
                # area_set = set()
                # print(best_x, best_y, best_w, best_h, best_area)
                cv2.rectangle(src_dilated, (x, y), (x + w, y + h), (255, 255, 255))
                for i in range(best_w):
                    for j in range(best_h):
                        # if flag:
                        #     break
                        # area_set.add(stats[labels[y + j, x + i]][-1])
                        if stats[labels[y + j, x + i]][-1] == best_area:
                            sky_line_num = labels[y + j, x + i]
                            # flag = 1
                # print(area_set)
                # print(sky_line_num)

    # cv2.imshow('1', src_dilated)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 检测天际线识别
    if best_x == 0 and best_y == 0 and best_area == 0:
        return img_src

    
    # 天际线分割
    black_img = np.zeros((hight, width, 3), dtype=np.uint8)

    threshold = [hight] * width
    low_x, low_y = 0, 0

    for i in range(hight):
        for j in range(width):
            if labels[i, j] == sky_line_num:

                threshold[j] = min(threshold[j], i)
                if threshold[j] > low_y:
                    low_x, low_y = j, threshold[j]
                # black_img[i, j] = [0, 0, 255]

    for i in range(width):
        if not (best_x <= i <= best_x + best_w):
            threshold[i] = best_y

    for i in range(width):
        black_img[0:threshold[i], i] = (255, 255, 255)
        black_img[threshold[i]: hight, i] = (0, 0, 0)

    # cv2.imshow('1', black_img)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 从闪电图片中分割闪电
    img_thunder = cv2.imread(thunder_path)
    thunder_canny = cv2.Canny(img_thunder, 100, 150)

    thunder_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thunder_dilated = cv2.dilate(thunder_canny, thunder_kernel)

    d_h, d_w = thunder_dilated.shape
    thunder_dilated = cv2.resize(thunder_dilated, ((best_y + best_h), int((best_y + best_h) * d_w / d_h)))


    # 图片融合

    # 原图片roi
    target_h, target_w = thunder_dilated.shape
    roi_x = max(0, low_x - target_w // 2)
    roi_y = 0
    roi_h = target_h
    roi_w = target_w
    print(roi_x, roi_y, roi_w, roi_h)
    black_img = cv2.cvtColor(black_img, cv2.COLOR_BGR2GRAY)
    roi_mask = black_img[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w]
    roi = img_src[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w, :]
    roi_mask = cv2.bitwise_not(roi_mask)

    # 防止 roi 越界
    true_h, true_w = roi_mask.shape
    thunder_dilated = cv2.resize(thunder_dilated, (true_w, true_h))

    final_mask = cv2.bitwise_or(roi_mask, thunder_dilated)
    roi_mask = cv2.bitwise_not(roi_mask)
    final_mask = cv2.bitwise_and(final_mask, roi_mask)

    test_thunder = img_thunder[:, :, :]
    # test_thunder = cv2.resize(test_thunder, ((best_y + best_h), int((best_y + best_h) * d_w / d_h)))
    test_thunder = cv2.resize(test_thunder, (true_w, true_h))
    thunder = cv2.bitwise_and(test_thunder, test_thunder, mask=final_mask)

    final_mask_inv = cv2.bitwise_not(final_mask)

    back_ground = cv2.bitwise_and(roi, roi, mask=final_mask_inv)

    fusion = cv2.add(thunder, back_ground)
    result_thunder = img_src[:, :, :]
    result_thunder[roi_y: roi_y + roi_h, roi_x: roi_x + roi_w, :] = fusion

    # cv2.imshow('1', result_thunder)
    # k = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return result_thunder

def transfer_color(src_img, target_img):
    transfer = color_transfer(src_img, target_img)
    return transfer


if __name__ == "__main__":
    # 19624_2020_12_29 14_34_28_79  19872_2020_12_29 15_38_55_12  20200_2020_12_29 14_25_20_7 47344_2020_12_29 14_17_55_83 47808_2020_12_29 14_26_53_97
    # 47268_2020_12_29 15_55_33_27  7280_2020_12_29 15_21_27_91
    # src_path = "../data/Rainy/7280_2020_12_29 15_21_27_91.jpg"

    # 11532_2020_12_30 13_39_51_88
    src_path = "../data/Cloud/14184_2020_12_30 13_26_17_35.jpg"

    # thunder_00000  thunder_00021
    thunder_path = "../data/Thunderstorm/thunder_00000.jpg"

    thunder_path_list = ["../data/Thunderstorm/thunder_00000.jpg", "../data/Thunderstorm/thunder_00005.jpg", "../data/Thunderstorm/thunder_00040.jpg",  
    "../data/Thunderstorm/thunder_00039.jpg", "../data/Thunderstorm/thunder_00056.jpg",  "../data/Thunderstorm/thunder_00404.jpg",  "../data/Thunderstorm/thunder_00413.jpg",
     "../data/Thunderstorm/thunder_00420.jpg",  "../data/Thunderstorm/thunder_00493.jpg"]

    color_path_list = ["../data/Thunderstorm/thunder_00102.jpg", "../data/Thunderstorm/thunder_00257.jpg", "../data/Thunderstorm/thunder_00273.jpg",
    "../data/Thunderstorm/thunder_00307.jpg",  "../data/Thunderstorm/thunder_00395.jpg",  "../data/Thunderstorm/thunder_00508.jpg"]
    
    num_thunder = len(thunder_path_list)
    num_color = len(color_path_list)

    src_path_folder = "../data/Cloud"

    save_path = "../data/Generated_thunder/"
    with os.scandir(src_path_folder) as folder:
        for f_img in folder:
            src_path = f_img.path
            print(src_path)
            src_name = f_img.name
            choice = random.randint(0, num_thunder - 1)
            thunder_path = thunder_path_list[choice]
            # print(thunder_path)
            try:
                img_fusion = generate_thunder(src_path, thunder_path)
            except Exception:
                continue
            choice = random.randint(0, num_color - 1)
            color_path = color_path_list[choice]
            # print(color_path)
            img_thunder = cv2.imread(color_path)
            img_fusion = transfer_color(img_thunder, img_fusion)


            # cv2.imshow('1', img_fusion)
            # k = cv2.waitKey(0)
            # cv2.destroyAllWindows()

            cv2.imwrite(save_path + src_name, img_fusion)


    img_fusion = generate_thunder(src_path, thunder_path)
    
    img_thunder = cv2.imread(thunder_path)
    img_fusion = transfer_color(img_thunder, img_fusion)

    cv2.imshow('1', img_fusion)
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()
