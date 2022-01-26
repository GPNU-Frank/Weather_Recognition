# from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
import numpy as np
import cv2
import math

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 150  # Canny edge detection low threshold
canny_hthreshold = 250  # Canny edge detection high threshold

# Hough transform parameters
rho = 1  # rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 180  # theta的范围
threshold = 100  # 累加器中的值高于它时才认为是一条直线
min_line_length = 100  # 线的最短长度，比这个短的都被忽略
max_line_gap = 50  # 两条直线之间的最大间隔，小于此值，认为是一条直线

def roi_mask(img, vertices):  # img是输入的图像，verticess是兴趣区的四个点的坐标（三维的数组）
    mask = np.zeros_like(img)  # 生成与输入图像相同大小的图像，并使用0填充,图像为黑色
    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count  # 如果 channel_count=3,则为(255,255,255)
    else:
        mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)#使用白色填充多边形，形成蒙板
    masked_img = cv2.bitwise_and(img, mask)#img&mask，经过此操作后，兴趣区域以外的部分被蒙住了，只留下兴趣区域的图像
    return masked_img

def draw_roi(img, vertices):
    cv2.polylines(img, vertices, True, [255, 0, 0], thickness=2)

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def my_draw_lines_v1(img, lines, color=[255, 0, 0], thickness=2):

    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            print(k)
            if abs(k) > 0.5:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    

def my_draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    mid_points_x = []
    mid_points_y = []
    k_lines = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            print(k)
            if abs(k) > 0.5:
                # print(k)
                # 计算每一条线的中点，筛选去离群的直线
                mid_points_x.append(int((x1 + x2) / 2))
                mid_points_y.append(int((y1 + y2) / 2))
                k_lines.append((x1, y1, x2, y2))
                # cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    # 计算中点平均值
    avg_mid_point_x = sum(mid_points_x) / len(mid_points_x)
    avg_mid_point_y = sum(mid_points_y) / len(mid_points_y)
    print(avg_mid_point_x, avg_mid_point_y)

    dist_thre = 250
    dist_left = 750
    dist_right = 750
    left_line = None
    right_line = None
    print(img.shape)
    for idx, (x1, y1, x2, y2) in enumerate(k_lines):
        if math.sqrt((mid_points_x[idx] - avg_mid_point_x) ** 2 + (mid_points_y[idx] - avg_mid_point_y) **2) < dist_thre:
            if mid_points_x[idx] < dist_left:
                dist_left = mid_points_x[idx]
                left_line = x1, y1, x2, y2
            if img.shape[1] - mid_points_x[idx] < dist_right:
                dist_right = img.shape[1] - mid_points_x[idx]
                right_line = x1, y1, x2, y2
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    # cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), color, thickness)
    # cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # lines = cv2.HoughLines(img, rho, theta, threshold, min_line_len, max_line_gap)
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)  # 函数输出的直接就是一组直线点的坐标位置（每条直线用两个点表示[x1,y1],[x2,y2]）
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)#生成绘制直线的绘图板，黑底
    # draw_lines(line_img, lines)
    my_draw_lines_v1(line_img, lines)
    # my_draw_lines(line_img, lines)
    # draw_lanes(line_img, lines)
    return line_img


# def my_draw_lines(img, lines, color=[0, 0, 255], thickness=2):
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []#用于存储左边和右边的直线
    for line in lines:#对直线进行分类
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img

    clean_lines(left_lines, 0.1)#弹出左侧不满足斜率要求的直线
    clean_lines(right_lines, 0.1)#弹出右侧不满足斜率要求的直线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]#提取左侧直线族中的所有的第一个点
    left_points = left_points + [(x2, y2) for line in left_lines for x1 ,y1, x2, y2 in line]#提取左侧直线族中的所有的第二个点
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]#提取右侧直线族中的所有的第一个点
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]#提取右侧侧直线族中的所有的第二个点

    # left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
    # right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标

    left_vtx = calc_lane_vertices(left_points, 0, img.shape[0])#拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
    right_vtx = calc_lane_vertices(right_points, 0, img.shape[0])#拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标


    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)#画出直线
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)#画出直线

#将不满足斜率要求的直线弹出
def clean_lines(lines, threshold):
    slope=[]
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            slope.append(k)
    #slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)  # 计算斜率的平均值，因为后面会将直线和斜率值弹出
        diff = [abs(s - mean) for s in slope]  # 计算每条直线斜率与平均值的差值
        idx = np.argmax(diff)  # 计算差值的最大值的下标
        if diff[idx] > threshold:  # 将差值大于阈值的直线弹出
          slope.pop(idx)  # 弹出斜率
          lines.pop(idx)  # 弹出直线
        else:
          break


# 拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]  # 提取x
    y = [p[1] for p in point_list]  # 提取y
    fit = np.polyfit(y, x, 1)  # 用一次多项式x=a*y+b拟合这些点，fit是(a,b)
    fit_fn = np.poly1d(fit)  # 生成多项式对象a*y+b

    xmin = int(fit_fn(ymin))  # 计算这条直线在图像中最左侧的横坐标
    xmax = int(fit_fn(ymax))  # 计算这条直线在图像中最右侧的横坐标

    return [(xmin, ymin), (xmax, ymax)]

def process_an_image(img):
    roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])#目标区域的四个点坐标，roi_vtx是一个三维的数组
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)#图像转换为灰度图
    blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)#使用高斯模糊去噪声


    med_val = np.median(img) 
    lower = int(max(0, 0.7 * med_val))
    upper = int(min(255, 1.3 * med_val))
    edges = cv2.Canny(blur_gray, lower, upper)#使用Canny进行边缘检测


    # edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)#使用Canny进行边缘检测

    plt.imshow(edges)
    plt.show()
    # roi_edges = roi_mask(edges, roi_vtx)#对边缘检测的图像生成图像蒙板，去掉不感兴趣的区域，保留兴趣区
    
    roi_edges = edges
    plt.imshow(roi_edges)
    plt.show()

    # lines = cv2.hough_lines(roi_image, rho, theta, threshold, min_line_lenght, max_line_gap)
    # copy_img = np.copy(img)
    # draw_lines(copy_img, lines, [0, 0, 255], 6)
    # plt.imshow(copy_img)
    # plt.show()

    line_img = hough_lines(roi_edges, rho, theta, threshold, min_line_length, max_line_gap)  # 使用霍夫直线检测，并且绘制直线
    res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)  # 将处理后的图像与原图做融合
    return res_img




if __name__ == "__main__":
    # img_path = "../data_split_v5/train/Cloud/17424_2021_01_25 12_03_57_97_0.jpg"
    # img_path = "../data_split_v5/train/Cloud/19724_2021_01_25 11_47_30_58_0.jpg"
    # img_path = "../data_split_v5/train/Cloud/22740_2021_01_25 11_57_44_73_0.jpg"
    # img_path = "../data_split_v5/train/Cloud/29640_2021_01_05 11_02_35_98.jpg"
    # img_path = "../data_split_v5/train/Cloud/30428_2021_01_05 10_27_31_5.jpg"  # 有遮挡
    # img_path = "../data_split_v5/train/Cloud/36564_2021_01_05 11_09_28_42.jpg"
    # img_path = "../data_split_v5/train/Cloud/8348_2020_12_30 13_18_47_71.jpg"
    # img_path = "../data_split_v5/train/Snow/544_2021_01_15 10_20_35_5_0.jpg"
    # img_path = "../data_split_v5/train/Snow/3844_2021_01_15 09_42_00_23_0.jpg"
    # img_path = "../data_split_v5/train/Rainy/17140_2020_12_29 15_05_40_39.jpg"
    # img_path = "../data_split_v5/train/Rainy/46704_2020_12_29 14_21_01_40.jpg"
    # img_path = "../data_split_v5/train/Fog/8852_2021_01_22 15_38_54_37.jpg"
    # img_path = "../data_split_v5/train/Fog/7144_2021_01_22 15_43_06_58.jpg"
    # img_path = "../data_split_v5/train/Fog/7144_2021_01_22 15_50_49_87.jpg"
    # img_path = "../data_split_v5/train/Sunny/2260_2020_12_29 16_47_24_20.jpg"
    # img_path = "../data_split_v5/train/Sunny/13448_2020_12_29 16_51_41_26.jpg"
    # img_path = "../data_split_v5/train/Sunny/13132_2020_12_30 10_55_08_25.jpg"
    # img_path = "../data_split_v5/train/Sunny/11808_2020_12_29 16_28_55_1.jpg"
    # img_path = "../data_split_v5/train/Snow/1368_2021_01_15 10_35_36_11_0.jpg"
    # img_path = "../data_split_v5/train/Sunny/11472_2020_12_29 16_20_09_8.jpg"
    # img_path = "../data_split_v5/train/Sunny/46668_2020_12_30 10_30_49_16.jpg"
    img_path = "../data_split_v6/train/Sunny/20128_2020_12_30 10_56_02_14.jpg"
    img = cv2.imread(img_path)
    res_img = process_an_image(img)
    plt.imshow(res_img)
    plt.show()
