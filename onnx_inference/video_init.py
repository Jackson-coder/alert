'''
Description: 
Version: 
Author: Jackson-coder
Date: 2022-08-06 14:56:27
LastEditors: Jackson-coder
LastEditTime: 2022-08-06 15:37:59
'''

import cv2
import numpy as np

def getHorizonSlope(cur_frame):
    image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2HSV)
    image = cv2.inRange(image,np.array([0,0,0]),np.array([210,255,86]))
    cv2.bitwise_not(image,image)
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.dilate(image, kernel, iterations = 1)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(cur_frame,contours,-1,(0,0,255),3)

    kx = 10000

    for c in contours:

        if cv2.contourArea(c) < 20000 or cv2.contourArea(c) > 50000:
            continue
        # 找到边界坐标
        x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 找面积最小的矩形
        rect = cv2.minAreaRect(c)
        # 得到最小矩形的坐标
        box = cv2.boxPoints(rect)

        # 标准化坐标到整数
        box = np.int0(box)
        # 画出边界
        cv2.drawContours(cur_frame, [box], 0, (0, 0, 255), 3)

        kx = rect[2] if abs(rect[2])<abs(kx) else kx

    cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    cv2.imwrite("demo1.jpg", image)
    cv2.imwrite("demo2.jpg", cur_frame)
    return kx

def getVerticalSlope(pose_results, kpt_thr):

    ky = []

    for pose in pose_results:
        flag = True
        for p in pose:
            flag = bool(p[2] > kpt_thr)
            if flag is False:
                break

        # 完整姿态
        if flag is False:
            continue

        ankle = (pose[15] + pose[16])/2
        hip = (pose[11] + pose[12])/2

        ky.append((ankle[0]-hip[0])/(ankle[1]-hip[1]))
        
    return np.mean(ky)
    