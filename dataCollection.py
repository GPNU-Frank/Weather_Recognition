#coding:utf-8

import cv2
import os
import logging
import requests
import time
import json

logging.basicConfig(filename='./dataCollection.log',filemode='a+',level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

root = './data/'
if not os.path.exists(root):
    os.makedirs(root)

# video_full_path =["http://alipull.lngsyg.cn/lngro/0f9862db5531a959e257c4619df5088b.flv?onlyvideo=1", #沈吉高速
#                    #"http://alipull.lngsyg.cn/lngro/7da872f6c2ee6bdd6d6a280966c98f55.flv?onlyvideo=1", #沈阳营口立交
#                    "http://alipull.lngsyg.cn/lngro/hd-8EBC5CFDF352B7CF2B695C4A9D07A9CE.flv?onlyvideo=1",#辽宁葫芦岛
#                    "http://alipull.lngsyg.cn/lngro/hd-90FFA72CC9C7BE7D96B236D2FF1EF3C8.flv?onlyvideo=1", #丹阜高速
#                    "http://alivspull.runoneapp.com/runone/624845e8129f11ebac7c0242ac110004NCB1.flv",#江西福银高速
#                   #"http://alivspull.runoneapp.com/runone/hd-18E52793EC5A137CFA7CCE04244E1A72.flv", #福银高速昌九段K6
#                    #"http://alivspull.runoneapp.com/runone/hd-4CD4869E017C83777E9959669ACB9278.flv",#福银高速昌九段K11
#                    #"http://alivspull.runoneapp.com/runone/hd-9BFA317425D9A563D633A9FFC1A188CD.flv", #福银高速艾城站
#                    "https://runpull.runoneapp.com/runone/655a8aa185f47138b806fc390baee930.flv?t=5fe99313&k=60e05601cdbf5960ad40f3ce2a60bd31", #广河高速广州段
#                    "https://runpull.runoneapp.com/runone/hd-9C98EFCD33A3DA605BF2650659961A17.flv?t=5fe993d1&k=3d0046af55fbcf8e854ef3d5a93bd758",#广河高速广州段
#                    "https://runpull.runoneapp.com/runone/hd-F2E65CF7AC937A9A960FEA0C724CEC98.flv?t=5fe9945b&k=4cedd53ba7e8be7b2131bd263059df64",
#                    "https://runpull.runoneapp.com/runone/hd-1F37E6EF3467839BB59793ED4A7B3935.flv?t=5fe99508&k=35fdeedf70b84d8b7d4bc1077067daa6"#广中江高速棠下站
#                    ]
# location = ["40.59:124.59",
#             #"40.65:122.44",
#             "40.81:120.16",
#             "41.22:121.70",
#             "28.87:115.86",
#             #"29.02:115.79",
#             #"28.97:115.81",
#             #"29.12:115.76",
#             "23.25:113.45",
#             "23.30:113.57",
#             "23.49:113.93",
#             "22.69:113.02"
#             ]

def get_dict(path):
    with open(path, 'r') as f:
        s = f.read()
        camera_dict = json.loads(s)
        return camera_dict



if __name__ == '__main__':
    path = 'test/2020_12_29 13_00_03json.json'
    camera_dict = get_dict(path)

    print(len(camera_dict['url']))
    cfg_count = 100


    # exit()
    n = 0
    frame_count = 1
    success = True
    while frame_count < cfg_count:
        for i, (data, url) in enumerate(zip(camera_dict['camera_info'], camera_dict['url'])):
            url = url.rstrip()
        #get weather
            # location = str(data['data'][0]['latitude']) + ':' + str(data['data'][0]['longitude'])
            # url = "https://api.seniverse.com/v3/weather/now.json?key=SD7MXtNnduHgrMwzk&language=en&location={0}".format(location)
            # print(location)
            # #url = "https://api.seniverse.com/v3/pro/weather/grid/now.json?key=SD7MXtNnduHgrMwzk&location={0}".format("223.111.123.220:80")
            # req = requests.get(url, timeout=30)  # 请求连接
            # req_jason = req.json()  # 获取数据
            # print(req_jason)
            # now_trunk = req_jason['results']
            # weather = now_trunk[0]['now']['text']
            # print(weather)
        #get frame
            cap = cv2.VideoCapture(url)
            #width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #fps = cap.get(cv2.CAP_PROP_FPS)
            #print(fps)
            cap.isOpened()
            success, frame = cap.read()
            print('Read a new frame {0}: {1}'.format(frame_count, success))
            print(data['data'][0]['global_name'])

            # storeDir = root + weather
            storeDir = root + 'unclassify'
            if not os.path.exists(storeDir):
                os.makedirs(storeDir)
            
            now_time = time.strftime("%Y_%m_%d %H_%M_%S", time.localtime())

            if success:
                img_path = storeDir + '/{0}_{1}_{2}.jpg'.format(now_time, i, frame_count)
                cv2.imwrite(img_path, frame)
                logging.info('Read a new frame {0}_{1}: {2}'.format(i, frame_count, 'None'))
            else:
                a = 0
                pass
            cap.release()
        frame_count = frame_count + 1
        print("sleep")
        time.sleep(20)#20秒执行一次
        print("end sleep")