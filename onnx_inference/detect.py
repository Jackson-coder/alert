import cv2
import json
import math
import numpy as np


class Detector(object):
    '''Tutorial

    CFG_INIT = True
    JUDGE_INVADE = False

    mode = CFG_INIT

    if mode == CFG_INIT:
        detector = Detector()
        for frame_id in range(100):
            detector.cfg_init(...)
        print(detector.kx, detector.ky)

    elif mode == JUDGE_INVADE:
        detector = Detector(kx, ky)
        outputFrame = detector.judge3DInvade(...)


    '''

    def __init__(self, kx=None, ky=None):
        self.__ky_buffer = []
        self.kx = kx
        self.ky = ky

    # def getHorizonSlope(self, cur_frame):
    #     image = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2HSV)
    #     image = cv2.inRange(image,np.array([0,0,0]),np.array([210,255,86]))
    #     cv2.bitwise_not(image,image)
    #     kernel = np.ones((5, 5), np.uint8)
    #     image = cv2.dilate(image, kernel, iterations = 1)
    #     contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    #     # cv2.drawContours(cur_frame,contours,-1,(0,0,255),3)

    #     kx = 10000

    #     for c in contours:

    #         if cv2.contourArea(c) < 20000 or cv2.contourArea(c) > 50000:
    #             continue
    #         # 找到边界坐标
    #         x, y, w, h = cv2.boundingRect(c)  # 计算点集最外面的矩形边界
    #         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #         # 找面积最小的矩形
    #         rect = cv2.minAreaRect(c)
    #         # 得到最小矩形的坐标
    #         box = cv2.boxPoints(rect)

    #         # 标准化坐标到整数
    #         box = np.int0(box)
    #         # 画出边界
    #         cv2.drawContours(cur_frame, [box], 0, (0, 0, 255), 3)

    #         kx = rect[2] if abs(rect[2])<abs(kx) else kx

    #     cv2.drawContours(image, contours, -1, (255, 0, 0), 1)
    #     cv2.imwrite("demo1.jpg", image)
    #     cv2.imwrite("demo2.jpg", cur_frame)
    #     return kx

    def getHorizonSlope(self, json_file, scale=1):
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            # alert1 = json_data['shapes'][0]['points']
            # alert2 = json_data['shapes'][1]['points']

            step = json_data['step']

            for i in range(len(step)):
                step[i] = [int(step[i][0]/scale), int(step[i][1]/scale)]

            point1 = point2 = [0, 0]

            for point in step:
                if point[1] > point1[1]:
                    point2 = point1
                    point1 = point
                elif point[1] > point2[1]:
                    point2 = point

            # print(point1, point2)

            return (point1[1]-point2[1])/(point1[0]-point2[0])

    # def getVerticalSlope(self, pose_results, kpt_thr):

    #     ky = []

    #     for pose in pose_results:
    #         flag = True
    #         for p in pose:
    #             flag = bool(p[2] > kpt_thr)
    #             if flag is False:
    #                 break

    #         # 完整姿态
    #         if flag is False:
    #             continue

    #         ankle = (pose[15] + pose[16])/2
    #         hip = (pose[11] + pose[12])/2

    #         ky.append((ankle[0]-hip[0])/(ankle[1]-hip[1]))

    #     return np.mean(ky)

    def getVerticalSlope(self, pose, kpt_thr):

        # flag = True
        # for p in pose:
        #     flag = bool(p[2] > kpt_thr)
        #     if flag is False:
        #         break

        # # 不是完整姿态
        # if flag is False:
        #     return


        if pose[15][2] > kpt_thr and pose[16][2] > kpt_thr and pose[11][2] > kpt_thr and pose[12][2] > kpt_thr:
            ankle = (pose[15] + pose[16])/2
            hip = (pose[11] + pose[12])/2
            
            return (ankle[0]-hip[0])/(ankle[1]-hip[1])
        else:
            return


    def getLine(self, a, b):
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

    def getDist_P2L_V1(self, P, a, b):
        """计算点到直线的距离
            P：定点坐标
            a：直线a点坐标
            b：直线b点坐标
        """
        # 求直线方程
        A, B, C = self.getLine(a, b)
        # 代入点到直线距离公式
        distance = (A*P[0]+B*P[1]+C)/math.sqrt(A*A+B*B)

        return distance

    # ***** 点到直线的距离:P到AB的距离*****

    def getDist_P2L_V2(self, P, k, P0):
        """计算点到直线的距离
            P：定点坐标
            k：直线斜率
            P0：直线上一点
        """
        distance = abs(k*P[0]-P[1]+P0[1]-k*P0[0])/math.sqrt(k*k+1)

        return distance

    def getCrossPoint(self, k, P, a, b):
        """计算直线与直线的交点
            k: 直线1 斜率
            P：直线1 定点坐标
            a：直线2 a点坐标
            b：直线2 b点坐标
        """
        b0 = P[1] - k*P[0]
        A, B, C = self.getLine(a, b)
        x = -(B*b0+C)/(A+B*k+1e-10)
        y = -k*(B*b0+C)/(A+B*k+1e-10)+b0

        return [int(x), int(y)]

    def judgeCross(self, k, P, a, b):
        """判断直线与线段是否相交
            k: 直线1 斜率
            P：直线1 定点坐标
            a：线段2 a点坐标
            b：线段2 b点坐标
        """
        crossPoint = self.getCrossPoint(k, P, a, b)
        if (crossPoint[0]-a[0])*(crossPoint[0]-b[0]) <= 0 and (crossPoint[1]-a[1])*(crossPoint[1]-b[1]) <= 0:
            return crossPoint, True
        else:
            return crossPoint, False

    def getCrossPoints(self, json_file='/home/lyh/mmpose/tests/data/coco/1.json', kx=0, P=[960, 540, 1], scale=1.5):
        """寻找交点
            json_file: 分割数据存储文件
            k：斜率
            P：定点坐标
            scale: 原图相对于处理图的倍率

            return:
                Points:[levelpoint1, levelpoint2, levelpoint3,
                    levelpoint4, verticalpoint1, verticalpoint2]
        """
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            # alert1 = json_data['shapes'][0]['points']
            # alert2 = json_data['shapes'][1]['points']

            alert1 = json_data['left baffle']
            alert2 = json_data['right baffle']

            for i in range(len(alert1)):
                alert1[i] = [int(alert1[i][0]/scale), int(alert1[i][1]/scale)]

            for i in range(len(alert2)):
                alert2[i] = [int(alert2[i][0]/scale), int(alert2[i][1]/scale)]

            # 水平交点
            Points = []
            for i in range(len(alert1)):
                flag = False
                crossPoint, flag = self.judgeCross(
                    kx, P, alert1[i], alert1[(i+1) % len(alert1)])
                if flag:
                    Points.append(crossPoint)

            for i in range(len(alert2)):
                flag = False
                crossPoint, flag = self.judgeCross(
                    kx, P, alert2[i], alert2[(i+1) % len(alert2)])
                if flag:
                    Points.append(crossPoint)

            Points.sort()

        return Points

    def getNearestCrossPoints(self, points, P):
        """寻找交点
            points: 水平交点
            P：定点坐标

            return:
                Points:[levelpoint1, levelpoint2]
        """

        # 负的最大
        max_c = -10000
        # 正的最小
        min_c = 10000

        point1 = point2 = [0, 0]

        for point in points:
                max_c, point1 = (point[0]-P[0], point) if point[0]-P[0] < 0 and point[0]-P[0] > max_c else (max_c, point1)
                min_c, point2 = (point[0]-P[0], point) if point[0]-P[0] > 0 and point[0]-P[0] < min_c else (min_c, point2)
        return point1, point2


    def judge2DborderIn(self, json_file='/home/lyh/mmpose/tests/data/coco/1.json', kx=0, P=[960, 540, 1], score_threshold=0.3, scale=1.5):
        """判断点是否在二维区域内部
            json_file: 分割数据存储文件
            P：目标点
            kx：水平线斜率
            score_threshold：姿态点置信度
            scale: 输入特征图相对于原始标注图像的缩放系数，>1为缩小

            return:
                True or False
        """
        Points = self.getCrossPoints(json_file, kx, P, scale)

        if P[2] < score_threshold:
            return True

        if len(Points) != 0:
            if P[0] > Points[0][0] and P[0] < Points[-1][0]:
                return True
            else:
                return False
        else:
            return False

    def judge2DfarBorderIn(self, json_file, pose, score_threshold=0.3, scale=1):
        """判断远处的点是否在二维区域内部
            json_file: 分割数据存储文件
            pose：目标点
            score_threshold：姿态点置信度
            scale: 输入特征图相对于原始标注图像的缩放系数，>1为缩小

            return:
                True or False
        """
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)

            alert1 = json_data['left baffle']
            alert2 = json_data['right baffle']

            left_point1 = [0,0]
            left_point2 = [0,0]
            for i in range(len(alert1)):
                alert1[i] = [int(alert1[i][0]/scale), int(alert1[i][1]/scale)]
                if alert1[i][0]>left_point1[0]:
                    left_point2 = left_point1
                    left_point1 = alert1[i]
                    
                elif alert1[i][0]>left_point2[0]:
                    left_point2 = alert1[i]

            right_point1 = [10000,10000]
            right_point2 = [10000,10000]
            for i in range(len(alert2)):
                alert2[i] = [int(alert2[i][0]/scale), int(alert2[i][1]/scale)]
                if alert2[i][0]<right_point1[0]:
                    right_point2 = right_point1
                    right_point1 = alert2[i]
                    
                elif alert2[i][0]<right_point2[0]:
                    right_point2 = alert2[i]


            # 判断线段与直线相交
            if pose[0]-left_point1[0] < 0 or pose[0] - right_point1[0] > 0:
                return True
            else:
                return False



    def judgeDirection(self, Points, score_threshold):
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

    def regionJudge(self, json_file, pose_point, mode='normal', scale=1):
        """判断是否是上扶梯远处的点
            json_file: 分割数据存储文件
            pose_point：目标点
            mode: 是否是正常视角
            scale: 输入特征图相对于原始标注图像的缩放系数，>1为缩小

            return:
                True or False
        """
        if mode=='normal':
            return True
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)

            step = json_data['step']

            for i in range(len(step)):
                step[i] = [int(step[i][0]/scale), int(step[i][1]/scale)]
            points = np.array([step], dtype=np.int32)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            vetuex1 = [10000,10000]
            vetuex2 = [10000,10000]
            for vetuex in box:
                if vetuex[1]<vetuex1[1]:
                    vetuex2 = vetuex1
                    vetuex1 = vetuex
                    
                elif vetuex[1]<vetuex2[1]:
                    vetuex2 = vetuex

            print(vetuex1, vetuex2)
            
            
            min_y1 = 1000
            min_y2 = 1000

            for i in range(len(step)):
                if abs((step[i][0]-vetuex1[0])/(step[i][1]-vetuex1[1]))<1 and step[i][1]<min_y1:
                    min_y1 = step[i][1]
                    point1 = step[i]
                if abs((step[i][0]-vetuex2[0])/(step[i][1]-vetuex2[1]))<1 and step[i][1]<min_y2:
                    min_y2 = step[i][1]
                    point2 = step[i]
            distance = self.getDist_P2L_V1(pose_point, point1, point2)

            print(distance)
            if distance > 0:
                return False
            else:
                return True
            
    def need_judge(self, json_file, pose_result, kpt_thr, scale=1):
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)

            step = json_data['step']

            max_yi = [0,0]
            min_yi = [10000,10000]

            for i in range(len(step)):
                step[i] = [int(step[i][0]/scale), int(step[i][1]/scale)]
                max_yi = step[i] if max_yi[1] < step[i][1] else max_yi
                min_yi = step[i] if min_yi[1] > step[i][1] else min_yi

            if pose_result[2] > kpt_thr:
                if pose_result[1] < min_yi[1] or pose_result[1] > max_yi[1]:
                    return False

            return True

    def cfg_init(self, json_file, output, kpt_thr):

        # pose_results = output[:, 6:]
        # pose_results = np.reshape(pose_results,(-1, 17, 3))
        # pose_scores = output[:, 4]

        pose_result = output[6:]
        pose_result = np.reshape(pose_result,(17, 3))
        pose_score = output[4]

        if self.kx == None:
            self.kx = self.getHorizonSlope(json_file)
            print('\n----------水平方向(角度)：',self.kx,'----------')

        ky = self.getVerticalSlope(pose_result, kpt_thr)
        print('ky is not None:', ky is not None)
        print('np.isnan(ky) == False:',np.isnan(ky) == False)
        if ky is not None and np.isnan(ky) == False:
            self.__ky_buffer.append(ky)


        if len(self.__ky_buffer)>10:
            self.ky = np.mean(self.__ky_buffer)
            print('------垂直方向（x/y）：',self.ky,'------')

        # if self.kx != None and self.ky != None:
            # print('\n----------水平方向(角度)：',self.kx,'----------')
            # print('------垂直方向（x/y）：',self.ky,'------')

    def judge3DInvade(self, output, json_file, kpt_thr, vis_frame, mode='normal', scale=1):
        '''判断是否发生3D入侵
            json_file: 图像分割结果

            kpt_thr: 姿态点置信阈值

            vis_frame: 输入视频
            
            scale: 处理输出的视频帧相对于原始视频帧的缩小倍数

            mode: 相机安装是否过低, 正常情况下(不过低)为normal

            return:
                vis_frame: 加入3D入侵警告的视频帧
        '''

        assert self.kx!=None and self.ky!=None

        # pose_results = output[:, 6:]
        # pose_results = np.reshape(pose_results,(-1, 17, 3))
        # pose_scores = output[:, 4]

        pose = output[6:]
        pose = np.reshape(pose,(17, 3))
        pose_score = output[4]

        flag = True
        
        for p in pose:
            # print('p=', p[2])
            flag = bool(p[2] > kpt_thr)
            if flag == False:
                # print('False')
                break

        # # 只识别面朝前或者面朝后的人
        # if direction != self.judgeDirection(pose, kpt_thr): # judgeDirection 将在第三版代码更新后去除，将使用跟踪位移方向判断人的移动方向
        #                                                 # 异常动作如 侧面歪头 和 侧面伸腿 将使用跟踪协助判断
        #     continue
        if pose_score < 0.3:
            return vis_frame, False

        if self.need_judge(json_file, pose[15], kpt_thr, scale=scale) == False or \
            self.need_judge(json_file, pose[16], kpt_thr, scale=scale) == False:
            return vis_frame, False

        # 踢腿出界
        if self.judge2DborderIn(json_file, P=pose[15], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
            cv2.circle(vis_frame, [int(pose[15][0]), int(pose[15][1])], 5, (0, 0, 255), 8)
            print('warning5: leg out of border')
            return vis_frame, True
        if self.judge2DborderIn(json_file, P=pose[16], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
            cv2.circle(vis_frame, [int(pose[16][0]), int(pose[16][1])], 5, (0, 0, 255), 8)
            print('warning5: leg out of border')
            return vis_frame, True
        
         # 头是否歪了
        Crookedhead = False
        if pose[0][2] > kpt_thr and \
            pose[11][2] > kpt_thr and \
                pose[12][2] > kpt_thr:
            hip = (pose[11] + pose[12])/2
            if abs((pose[0][0]-hip[0])/(pose[0][1]-hip[1])) > 0.35:
                Crookedhead = True
            else:
                Crookedhead = False


        # 不是完整姿态
        if flag is False:
            in_border = 0
            out_border = 0
            for p in pose:
                if self.judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=self.kx, scale=scale): 
                    in_border += 1
                elif not self.judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=self.kx, scale=scale):
                    out_border += 1

            # 全部出界
            if in_border == 0 and out_border != 0 and Crookedhead is True:
                cv2.circle(vis_frame, [int(pose[0][0]),
                        int(pose[0][1])], 5, (0, 0, 255), 8)
                print('warning1: out of border')
                return vis_frame, True
            # 内外都有，手在内部，头部歪曲
            elif self.judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=self.kx, scale=scale) == True and \
                self.judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=self.kx, scale=scale) == True and \
                    Crookedhead is True and out_border != 0 and in_border != 0:
                cv2.circle(vis_frame, [int(pose[0][0]),
                        int(pose[0][1])], 5, (0, 0, 255), 8)
                print('warning2: out of border')
                return vis_frame, True
            # 内外都有，手部伸展
            elif out_border != 0 and in_border != 0:
                if self.judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
                    if abs((pose[9][1]-pose[7][1])/(pose[9][0]-pose[7][0])) < 1 and abs((pose[5][1]-pose[7][1])/(pose[5][0]-pose[7][0])) < 1:
                        p = [int(pose[9][0]), int(pose[9][1])]
                        cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                        print('warning3: out of border')
                        return vis_frame, True
                elif self.judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
                    if abs((pose[10][1]-pose[8][1])/(pose[10][0]-pose[8][0])) < 1 and abs((pose[6][1]-pose[8][1])/(pose[6][0]-pose[8][0])) < 1:
                        p = [int(pose[10][0]), int(pose[10][1])]
                        cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                        print('warning4: out of border')
                        return vis_frame, True

            # elif out_border != 0 and in_border != 0:
            #     if self.judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False and pose[15][2]<0.5 and pose[16][2]<0.5:
            #         if abs((pose[9][1]-pose[7][1])/(pose[9][0]-pose[7][0])) < 2:
            #             p = [int(pose[9][0]), int(pose[9][1])]
            #             cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #             print('warning3: out of border')
            #     elif self.judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False and pose[15][2]<0.5 and pose[16][2]<0.5:
            #         if abs((pose[10][1]-pose[8][1])/(pose[10][0]-pose[8][0])) < 2:
            #             p = [int(pose[10][0]), int(pose[10][1])]
            #             cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #             print('warning4: out of border')
            #     elif pose[15][2]>0.5 and pose[16][2]<0.5:
            #         crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[15], scale=scale)
            #         if len(crossPoints) > 3:
            #             point1, point2 = self.getNearestCrossPoints(crossPoints, pose[15])
                    
            #             if (pose[9][0]-point1[0]) * (pose[9][0]-point2[0]) > 0 and pose[9][2]>0.5:
            #                 p = [int(pose[9][0]), int(pose[9][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')
            #             elif (pose[10][0]-point1[0]) * (pose[10][0]-point2[0]) > 0 and pose[10][2]>0.5:
            #                 p = [int(pose[10][0]), int(pose[10][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')

            #     elif pose[15][2]<0.5 and pose[16][2]>0.5:
            #         crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[16], scale=scale)
            #         if len(crossPoints) > 3:
            #             point1, point2 = self.getNearestCrossPoints(crossPoints, pose[16])
                        
            #             if (pose[9][0]-point1[0]) * (pose[9][0]-point2[0]) > 0 and pose[9][2]>0.5:
            #                 p = [int(pose[9][0]), int(pose[9][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')
            #             elif (pose[10][0]-point1[0]) * (pose[10][0]-point2[0]) > 0 and pose[10][2]>0.5:
            #                 p = [int(pose[10][0]), int(pose[10][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')
                
            #     elif pose[15][2]>0.5 and pose[16][2]>0.5:
            #         if pose[15][1] > pose[16][1]:
            #             crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[15], scale=scale)
            #         else:
            #             crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[16], scale=scale)

            #         if len(crossPoints) > 3:
            #             if pose[15][1] > pose[16][1]:
            #                 point1, point2 = self.getNearestCrossPoints(crossPoints, pose[15])
            #             else:
            #                 point1, point2 = self.getNearestCrossPoints(crossPoints, pose[16])

            #             if (pose[9][0]-point1[0]) * (pose[9][0]-point2[0]) > 0 and pose[9][2]>0.5:
            #                 p = [int(pose[9][0]), int(pose[9][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')
            #             elif (pose[10][0]-point1[0]) * (pose[10][0]-point2[0]) > 0 and pose[10][2]>0.5:
            #                 p = [int(pose[10][0]), int(pose[10][1])]
            #                 cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
            #                 print('warning6: out of border')

            return vis_frame, False

        # 完整姿态
        region_need_to_judge = self.regionJudge(json_file, pose[15], mode=mode, scale=1) and \
            self.regionJudge(json_file, pose[16], mode=mode, scale=1)

        if region_need_to_judge == False:
            if abs((pose[9][1]-pose[7][1])/(pose[9][0]-pose[7][0])) < 1 \
                    and abs((pose[5][1]-pose[7][1])/(pose[5][0]-pose[7][0])) < 1 \
                        and self.judge2DfarBorderIn(json_file, pose[9]):
                p = [int(pose[9][0]), int(pose[9][1])]
                cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                print('warning6: out of border')
                return vis_frame, True
            elif abs((pose[10][1]-pose[8][1])/(pose[10][0]-pose[8][0])) < 1 \
                     and abs((pose[6][1]-pose[8][1])/(pose[6][0]-pose[8][0])) < 1 \
                         and self.judge2DfarBorderIn(json_file, pose[10]):
                p = [int(pose[10][0]), int(pose[10][1])]
                cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                print('warning6: out of border')
                return vis_frame, True
            elif Crookedhead == True  and self.judge2DfarBorderIn(json_file, pose[0]):
                p = [int(pose[0][0]), int(pose[0][1])]
                cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                print('warning6: out of border')
                return vis_frame, True
            return vis_frame, False

        # 水平交点
        if pose[15][1] > pose[16][1]:
            crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[15], scale=scale)
        else:
            crossPoints = self.getCrossPoints(json_file, kx=self.kx, P=pose[16], scale=scale)
        # 得到最近的两个水平交点

        # if len(crossPoints) != 0:
        if len(crossPoints) > 3:
            if pose[15][1] > pose[16][1]:
                point1, point2 = self.getNearestCrossPoints(crossPoints, pose[15])
            else:
                point1, point2 = self.getNearestCrossPoints(crossPoints, pose[16])

            for point in crossPoints:
                cv2.circle(vis_frame, point, 5, (255, 0, 0), 8)
            parallelLineDistance = self.getDist_P2L_V2(
                point1, 1/self.ky, point2)

            shoulder = (pose[5] + pose[6]) / 2
            hip = (pose[11] + pose[12]) / 2
            tolerable_eer_thr = abs(hip[1] - shoulder[1]) *0.5

            for p in pose:
                distance1 = self.getDist_P2L_V2(p, 1/self.ky, point1)
                distance2 = self.getDist_P2L_V2(p, 1/self.ky, point2)
                
                if abs(distance1+distance2-parallelLineDistance) > tolerable_eer_thr:
                    print('warning: out of border')
                    cv2.circle(vis_frame, [int(pose[0][0]), int(
                        pose[0][1])], 10, (0, 0, 255), 8)
                    return vis_frame, True

        return vis_frame, False



if __name__ == "__main__":
    detector = Detector()
    # print(detector.getHorizonSlope("E:\\alert\demo2\seg_result.json"))
    # pose_result = np.array([[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5],[1,2,0.5]])
    # ky = detector.getVerticalSlope(pose_result, 0.3)
    json_file = "E:\\alert\demo\demo1.json"
    # output = [5,6,5,7,1,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5,1,2,0.5]
    # kpt_thr = 0.3
    # detector.cfg_init(json_file, output, kpt_thr)
    pose_point = [557, 439]
    detector.regionJudge(json_file, pose_point, 'unnormal')
