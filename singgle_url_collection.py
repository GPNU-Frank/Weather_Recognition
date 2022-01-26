import argparse
import cv2
import time
import os
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 参数: 链接  天气分类  取多少帧  每帧间隔
    parser.add_argument('-u', '--url', type=str)
    parser.add_argument('-w', '--weather', type=str)
    parser.add_argument('-f', '--frame-num', default=100, type=int)
    # parser.add_argument('-g', '--gap', default=20, type=int)
    parser.add_argument('-g', '--gap', default=60, type=int)

    # 取连续帧用
    parser.add_argument('-s', '--sequence', default=5, type=int)

    args = parser.parse_args()

    frame_count = 0
    print(args)

    # 保存的目录
    save_dir = './data/' + args.weather
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    while frame_count < args.frame_num:
        for i in range(args.sequence):
            try:
                cap = cv2.VideoCapture(args.url)
                cap.isOpened()
            except Exception:
                break
            success, frame = cap.read()

            now_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())

            if success:
                img_path = save_dir + '/{0}_{1}_{2}_{3}.jpg'.format(os.getpid(), now_time, frame_count, i)
                cv2.imwrite(img_path, frame)

        cap.release()
        frame_count = frame_count + 1

        time.sleep(args.gap)
    
