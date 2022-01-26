
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from skimage.morphology import remove_small_objects
im1=cv2.imread(r'G:\vscode_workspace\Weather_Recognition\test\misclassify_04_22_v6_densenet_transfer\1042_Sunny_Cloud.jpg')
b,g,r=np.double(cv2.split(im1))
shadow_ratio = (4/np.pi)*np.arctan2((b-g),(b+g)) #mutiply 4/pi is to ensure value[0,1]
shadow_mask=shadow_ratio>0.5
#cv2.imshow("shadow_mask",np.uint8(shadow_mask*255))
shadow_mask[:5,:]=0
shadow_mask[-5:,:]=0
shadow_mask[:,:5]=0
shadow_mask[:,-5:]=0#边界上的值=0
#cv2.imshow("shadow_mask1",np.uint8(shadow_mask*255))
shadow_mask=remove_small_objects(shadow_mask, min_size=100, connectivity=3)
# opencv 中没有matlab 中类似bwareaopen的函数，二值图像面积开运算
cv2.imshow("shadow_mask1",np.uint8(shadow_mask*255))
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel[1,0]=0
kernel[3,0]=0
kernel[1,4]=0
kernel[3,4]=0
shadow_mask1=np.uint8(shadow_mask*1)
mask=cv2.dilate(shadow_mask1,kernel)-shadow_mask1
#cv2.imshow("boundary",np.uint8(mask*255))
#substarct shadow_mask is to get boundary
#get boundary
[row,col]=np.where(mask==1)
#for i in range(len(row)-1):
#    cv2.line(im1,(col[i],row[i]),(col[i+1],row[i+1]),(0,0,255),1)
im1[row,col,:]=im1[40,40,:]
cv2.imshow("original-shadow",im1)
cv2.waitKey(0)
