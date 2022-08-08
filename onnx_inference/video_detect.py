'''
Description: 
Version: 
Author: Jackson-coder
Date: 2022-08-06 15:43:14
LastEditors: Jackson-coder
LastEditTime: 2022-08-07 19:31:05
'''
import cv2
import numpy as np
import json
import math

def getLine(a, b):
    """计算直线方程
        A*x+B*y+C=0
        a：直线a点坐标
        b：直线b点坐标
    """
    A = a[1]-b[1]
    B = b[0]-a[0]
    C = a[0]*b[1]-a[1]*b[0]

    return A, B, C


# ***** 点到直线的距离:P到AB的距离*****
# P为线外一点，AB为线段两个端点
def getDist_P2L_V1(P, a, b):
    """计算点到直线的距离
        P：定点坐标
        a：直线a点坐标
        b：直线b点坐标
    """
    # 求直线方程
    A, B, C = getLine(a, b)
    # 代入点到直线距离公式
    distance = (A*P[0]+B*P[1]+C)/math.sqrt(A*A+B*B)

    return distance

# ***** 点到直线的距离:P到AB的距离*****


def getDist_P2L_V2(P, k, P0):
    """计算点到直线的距离
        P：定点坐标
        k：直线斜率
        P0：直线上一点
    """
    distance = abs(k*P[0]-P[1]+P0[1]-k*P0[0])/math.sqrt(k*k+1)

    return distance


def getCrossPoint(k, P, a, b):
    """计算直线与直线的交点
        k: 直线1 斜率
        P：直线1 定点坐标
        a：直线2 a点坐标
        b：直线2 b点坐标
    """
    b0 = P[1] - k*P[0]
    A, B, C = getLine(a, b)
    x = -(B*b0+C)/(A+B*k)
    y = -k*(B*b0+C)/(A+B*k)+b0

    return [int(x), int(y)]


def judgeCross(k, P, a, b):
    """判断直线与线段是否相交
        k: 直线1 斜率
        P：直线1 定点坐标
        a：线段2 a点坐标
        b：线段2 b点坐标
    """
    crossPoint = getCrossPoint(k, P, a, b)
    if (crossPoint[0]-a[0])*(crossPoint[0]-b[0]) <= 0 and (crossPoint[1]-a[1])*(crossPoint[1]-b[1]) <= 0:
        return crossPoint, True
    else:
        return crossPoint, False


def getCrossPoints(json_file='/home/lyh/mmpose/tests/data/coco/1.json', kx=0, P=[960, 540, 1], scale=1.5):
    """寻找交点
        json_file: 分割数据存储文件
        k：斜率
        P：定点坐标
        scale: 原图相对于处理图的倍率

        return:
            Points:[levelpoint1, levelpoint2, levelpoint3, levelpoint4, verticalpoint1, verticalpoint2]
    """
    with open(json_file, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        alert1 = json_data['shapes'][0]['points']
        alert2 = json_data['shapes'][1]['points']

        for i in range(len(alert1)):
            alert1[i] = [int(alert1[i][0]/scale), int(alert1[i][1]/scale)]

        for i in range(len(alert2)):
            alert2[i] = [int(alert2[i][0]/scale), int(alert2[i][1]/scale)]

        # 水平交点
        Points = []
        for i in range(len(alert1)):
            flag = False
            crossPoint, flag = judgeCross(
                kx, P, alert1[i], alert1[(i+1) % len(alert1)])
            if flag:
                Points.append(crossPoint)

        for i in range(len(alert2)):
            flag = False
            crossPoint, flag = judgeCross(
                kx, P, alert2[i], alert2[(i+1) % len(alert2)])
            if flag:
                Points.append(crossPoint)

        Points.sort()

    return Points


def judge2DborderIn(json_file='/home/lyh/mmpose/tests/data/coco/1.json', kx=0, P=[960, 540, 1], score_threshold=0.3, scale=1.5):
    """判断点是否在二维区域内部
        json_file: 分割数据存储文件
        P：目标点
        kx：水平线斜率
        score_threshold：姿态点置信度
        scale: 输入特征图相对于原始标注图像的缩放系数，>1为缩小

        return:
            True or False
    """
    Points = getCrossPoints(json_file, kx, P, scale)

    if P[2] < score_threshold:
        return True

    if len(Points) != 0:
        if P[0] > Points[0][0] and P[0] < Points[-1][0]:
            return True
        else:
            return False
    else:
        return False


def judgeDirection(Points, score_threshold):
    '''判断人体方向
        Points: 人体关节点
        score_threshold：关节点阈值

        return:
            ‘forward' or 'backward'
    '''
    # 投票定方向
    forward_count = 0
    backward_count = 0
    a = 1
    for i in range(1, 8):
        # print(Points[a], ' ', Points[a+1])
        if Points[a][2] > score_threshold and \
            Points[a+1][2] > score_threshold and \
                Points[a][0] - Points[a+1][0] > 0:
            forward_count += 1
        elif Points[a][2] > score_threshold and \
            Points[a+1][2] > score_threshold and \
                Points[a][0] - Points[a+1][0] < 0:
            backward_count += 1
        a += 2

    if forward_count > backward_count:
        return 'forward'
    else:
        return 'backward'


def judge3DInvade(json_file, pose_results, pose_scores, direction, kpt_thr, vis_frame, tolerable_eer_thr, kx, ky, scale):
    '''判断是否发生3D入侵
        json_file: 图像分割结果

        pose_results: 姿态估计结果，具体格式如下：
                        0:'nose',
                        1:'left_eye',
                        2:'right_eye',
                        3:'left_ear',
                        4:'right_ear',
                        5:'left_shoulder',
                        6:'right_shoulder',
                        7:'left_elbow',
                        8:'right_elbow',
                        9:'left_wrist',
                        10:'right_wrist',
                        11:'left_hip',
                        12:'right_hip',
                        13:'left_knee',
                        14:'right_knee',
                        15:'left_ankle',
                        16:'right_ankle',
                        
        pose_scores: 人体姿态框置信度

        direction:识别对象行进方向

        kpt_thr: 姿态点置信阈值

        vis_frame: 输入视频

        tolerable_eer_thr: 针对完整17关节点识别对象的边界容错阈值

        kx:水平线斜率
        
        ky:垂直线斜率
        
        scale: 处理输出的视频帧相对于原始视频帧的缩小倍数

        return:
            vis_frame: 加入3D入侵警告的视频帧
    '''
    for pose, pose_score in zip(pose_results, pose_scores):
        flag = True
        for p in pose:
            flag = bool(p[2] > kpt_thr)
            if flag is False:
                break

        # 只识别面朝前或者面朝后的人
        if direction != judgeDirection(pose, kpt_thr): # judgeDirection 将在第三版代码更新后去除，将使用跟踪位移方向判断人的移动方向
                                                        # 异常动作如 侧面歪头 和 侧面伸腿 将使用跟踪协助判断
            continue
        if pose_score < 0.3:
            continue

        # 不是完整姿态
        if flag is False:
            # 头是否歪了
            Crookedhead = False
            if pose[0][2] > kpt_thr and \
                pose[1][2] > kpt_thr and \
                    pose[2][2] > kpt_thr:
                eye = (pose[1] + pose[2])/2
                if abs((pose[0][0]-eye[0])/(pose[0][1]-eye[1])) > 0.3:
                    Crookedhead = True
                else:
                    Crookedhead = False

            in_border = 0
            out_border = 0
            for p in pose:
                if judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=kx, scale=scale): 
                    in_border += 1
                elif not judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=kx, scale=scale):
                    out_border += 1

            # 全部出界
            if in_border == 0 and out_border != 0 and Crookedhead is True:
                cv2.circle(vis_frame, [int(pose[0][0]),
                           int(pose[0][1])], 5, (0, 0, 255), 8)
                print('warning1: out of border')
            # 内外都有，手在内部，头部歪曲
            elif judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=kx, scale=scale) == True and \
                judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=kx, scale=scale) == True and \
                    Crookedhead is True and out_border != 0 and in_border != 0:
                cv2.circle(vis_frame, [int(pose[0][0]),
                           int(pose[0][1])], 5, (0, 0, 255), 8)
                print('warning2: out of border')
            # 内外都有，手部伸展
            elif out_border != 0 and in_border != 0:
                if judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=kx, scale=scale) == False:
                    if abs((pose[9][1]-pose[7][1])/(pose[9][0]-pose[7][0])) < 2:
                        p = [int(pose[9][0]), int(pose[9][1])]
                        cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                        print('warning3: out of border')
                elif judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=kx, scale=scale) == False:
                    if abs((pose[10][1]-pose[8][1])/(pose[10][0]-pose[8][0])) < 2:
                        p = [int(pose[10][0]), int(pose[10][1])]
                        cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                        print('warning4: out of border')

            continue

        # 完整姿态
        # 四个水平交点
        crossPoints = getCrossPoints(P=pose[15])

        if len(crossPoints) != 0:
            # print(crossPoints)
            for point in crossPoints:
                cv2.circle(vis_frame, point, 5, (255, 0, 0), 8)
            parallelLineDistance = getDist_P2L_V2(
                crossPoints[1], 1/ky, crossPoints[2])
            for p in pose:
                distance1 = getDist_P2L_V2(p, 1/ky, crossPoints[1])
                distance2 = getDist_P2L_V2(p, 1/ky, crossPoints[2])
                if abs(distance1+distance2-parallelLineDistance) > tolerable_eer_thr:
                    # print(abs(distance1+distance2-parallelLineDistance))
                    print('warning: out of border')
                    cv2.circle(vis_frame, [int(pose[0][0]), int(
                        pose[0][1])], 10, (0, 0, 255), 8)

    return vis_frame
