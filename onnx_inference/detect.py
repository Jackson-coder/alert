import cv2
import json
import math
import numpy as np

class Detector(object):
    '''Tutorial

    detector = Detector(conf_file, mode='normal')
    invade_flag = detector.judge3DInvade(kpts, kpt_score, im0)  #debug
    invade_flag = detector.judge3DInvade(kpts, kpt_score)  #release

    '''

    def __init__(self, json_file, mode='normal', k_thr=0.1, scale=1):
        """
         
            json_file: 图像分割结果文件
            mode: 相机安装是否过低, 正常情况下(不过低)为normal
            k_thr: 构成水平线的相邻点的斜率范围
            scale: 原图相对于处理图的倍率,输入特征图相对于原始标注图像的缩放系数，>1为缩小
        """
        with open(json_file, 'r', encoding='utf8')as fp:
            json_data = json.load(fp)
            self.step = json_data['step']
            self.alert1 = json_data['left baffle']
            self.alert2 = json_data['right baffle']
            self.floor_plate = json_data["floor plate"]
            self.larger_floor = json_data["larger_floor"]
            self.width = json_data["imgWidth"]
            self.height = json_data["imgHeight"]

            # self.farborder = json_data[]
            # self.farBorder = [[[585,426],[589,496]],[[668,425],[676,494]]]

            for i in range(len(self.alert1)):
                self.alert1[i] = [int(self.alert1[i][0]/scale), int(self.alert1[i][1]/scale)]

            for i in range(len(self.alert2)):
                self.alert2[i] = [int(self.alert2[i][0]/scale), int(self.alert2[i][1]/scale)]

            for i in range(len(self.step)):
                self.step[i] = [int(self.step[i][0]/scale), int(self.step[i][1]/scale)]

            for i in range(len(self.floor_plate)):
                self.floor_plate[i] = [int(self.floor_plate[i][0]/scale), int(self.floor_plate[i][1]/scale)]

            for i in range(len(self.larger_floor)):
                self.larger_floor[i] = [int(self.larger_floor[i][0]/scale), int(self.larger_floor[i][1]/scale)]

        self.mode = mode
        # 求直线方程
        if self.mode == 'unnormal':
            self.A, self.B, self.C = self.getBorder() 

        self.kx = self.getHorizonSlope(k_thr)

    def getHorizonSlope_v1(self):
        """计算水平斜率
        """
        point1 = point2 = [0, 0]

        for point in self.step:
            print(point)
            if point[1] > point1[1]:
                point2 = point1
                point1 = point
            elif point[1] > point2[1]:
                point2 = point
        
        kx = (point1[1]-point2[1])/(point1[0]-point2[0]+1e-10)
        print('\n----------水平方向(角度)：',kx,'----------')
        return kx

    def getHorizonSlope_v2(self):
        """计算水平斜率
        """
        point1 = [10000, 10000]
        point2 = [0, 0]

        for point in self.step:
            print(point)
            if point1[0] > point[0]:
                point1 = point
            elif point2[0] < point[0]:
                point2 = point
        
        kx = (point1[1]-point2[1])/(point1[0]-point2[0]+1e-10)
        print('\n----------水平方向(角度)：',kx,'----------')
        return kx
    
    def getHorizonSlope_v3(self):
        """计算水平斜率
        """
        step = np.array(self.step)
        rect = cv2.minAreaRect(step)
        box = cv2.boxPoints(rect)
        box_int = np.int0(box)

        point1 = point2 = [0, 0]

        for point in box:
            print(point)
            if point[1] > point1[1]:
                point2 = point1
                point1 = point
            elif point[1] > point2[1]:
                point2 = point
        
        kx = (point1[1]-point2[1])/(point1[0]-point2[0]+1e-10)
        print('\n----------水平方向(角度)：',kx,'----------')
        return kx

    def getHorizonSlope(self, k_thr=0.1):
        """计算水平斜率
        """
        p = [0, 0]

        for point in self.step:
            if point[1] > p[1]:
                p = point
        
        left = p
        right = p
        for point in self.step:
            if point[0] < left[0] and abs((point[1]-left[1])/(point[0]-left[0]))<k_thr:
                left = point
            if point[0] > right[0] and abs((point[1]-right[1])/(point[0]-right[0]))<k_thr:
                right = point
        
        print('\n----------',left,'.....',right,'----------')
        
        kx = (right[1]-left[1])/(right[0]-left[0]+1e-10)
        print('\n----------水平方向(角度)：',kx,'----------')
        return kx

    # def getVerticalSlope(self, pose, kpt_thr):
    #     if pose[15][2] > kpt_thr and pose[16][2] > kpt_thr and pose[11][2] > kpt_thr and pose[12][2] > kpt_thr:
    #         ankle = (pose[15] + pose[16])/2
    #         hip = (pose[11] + pose[12])/2
            
    #         return (ankle[0]-hip[0])/(ankle[1]-hip[1]+1e-10)
    #     else:
    #         return


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
        distance = (A*P[0]+B*P[1]+C)/math.sqrt(A*A+B*B+1e-10)

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

    def getCrossPoints(self, kx=0, P=[960, 540, 1]):
        """寻找交点
            k：斜率
            P：定点坐标

            return:
                Points:[levelpoint1, levelpoint2, levelpoint3,
                    levelpoint4, verticalpoint1, verticalpoint2]
        """
        # 水平交点
        Points = []
        for i in range(len(self.alert1)):
            flag = False
            crossPoint, flag = self.judgeCross(
                kx, P, self.alert1[i], self.alert1[(i+1) % len(self.alert1)])
            if flag:
                Points.append(crossPoint)

        for i in range(len(self.alert2)):
            flag = False
            crossPoint, flag = self.judgeCross(
                kx, P, self.alert2[i], self.alert2[(i+1) % len(self.alert2)])
            if flag:
                Points.append(crossPoint)

        Points.sort()
        print(Points)

        return Points
    

    def getLeftPoint(self, kx=0, P=[960, 540, 1]):
        """寻找交点
            k：斜率
            P：定点坐标

            return:
                Point
        """
        # 水平交点
        Points = []
        for i in range(len(self.alert1)):
            flag = False
            crossPoint, flag = self.judgeCross(
                kx, P, self.alert1[i], self.alert1[(i+1) % len(self.alert1)])
            if flag:
                Points.append(crossPoint)

        Points.sort()

        return Points


    def getRightPoint(self, kx=0, P=[960, 540, 1]):
        """寻找交点
            k：斜率
            P：定点坐标

            return:
                Point
        """
        # 水平交点
        Points = []

        for i in range(len(self.alert2)):
            flag = False
            crossPoint, flag = self.judgeCross(
                kx, P, self.alert2[i], self.alert2[(i+1) % len(self.alert2)])
            if flag:
                Points.append(crossPoint)

        Points.sort()

        return Points


    def getNearestCrossPoints(self, points, P, position='middle'):
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

        if position=='middle':
            for point in points:
                    max_c, point1 = (point[0]-P[0], point) if point[0]-P[0] < 0 and point[0]-P[0] > max_c else (max_c, point1)
                    min_c, point2 = (point[0]-P[0], point) if point[0]-P[0] > 0 and point[0]-P[0] < min_c else (min_c, point2)
            return point1, point2
        elif position=='right':
            for point in points:
                    max_c, point1 = (point[0]-P[0], point) if point[0]-P[0] < 0 and point[0]-P[0] > max_c else (max_c, point1)
                    min_c, point2 = (point[0]-P[0], point) if point[0]-P[0] < 0 and point[0]-P[0] < min_c else (min_c, point2)
            return point1, point2
        else:
            for point in points:
                    max_c, point1 = (point[0]-P[0], point) if point[0]-P[0] > 0 and point[0]-P[0] > max_c else (max_c, point1)
                    min_c, point2 = (point[0]-P[0], point) if point[0]-P[0] > 0 and point[0]-P[0] < min_c else (min_c, point2)
            return point1, point2


    def judge2DborderIn(self, kx=0, P=[960, 540, 1], score_threshold=0.3, error_thr=0):
        """判断点是否在二维区域内部
            P：目标点
            kx：水平线斜率
            score_threshold：姿态点置信度

            return:
                True or False
        """
        Points = self.getCrossPoints(kx, P)
        delta = 0
        if len(Points)>3:       
            if P[0] >= Points[0][0] and P[0] <= Points[-1][0]:   # point is in the middle
                # print('middle')
                point1, point2 = self.getNearestCrossPoints(Points, P, 'middle')
                delta = error_thr * abs(point1[0]-point2[0])
            elif P[0] > Points[-1][0]: 
                # print('right')
                point1, point2 = self.getNearestCrossPoints(Points, P, 'right')
                delta = error_thr * abs(point1[0]-point2[0])
            else:
                # print('left')
                point1, point2 = self.getNearestCrossPoints(Points, P, 'left') # point is in the left
                delta = error_thr * abs(point1[0]-point2[0])

        if P[2] < score_threshold:
            return True

        if len(Points) >3:
            if P[0] > Points[0][0] - delta and P[0] < Points[-1][0]  + delta:
                return True
            else:
                return False
        else:
            return True

    def judge2DfarBorderIn(self, pose, error_thr=0.15, score_threshold=0.3):
        """判断远处的点是否在二维区域内部
            pose：目标点
            score_threshold：姿态点置信度

            return:
                True or False
        """
        left_point1 = [0,0]
        left_point2 = [0,0]
        for i in range(len(self.alert1)):
            if self.alert1[i][0]>left_point1[0]:
                left_point2 = left_point1
                left_point1 = self.alert1[i]
                
            elif self.alert1[i][0]>left_point2[0]:
                left_point2 = self.alert1[i]

        right_point1 = [10000,10000]
        right_point2 = [10000,10000]
        for i in range(len(self.alert2)):
            if self.alert2[i][0]<right_point1[0]:
                right_point2 = right_point1
                right_point1 = self.alert2[i]
                
            elif self.alert2[i][0]<right_point2[0]:
                right_point2 = self.alert2[i]

        delta = error_thr * abs(left_point1[0]-right_point1[0])
        # 判断线段与直线相交
        if pose[0]-left_point1[0] + delta < 0 or pose[0] - right_point1[0] - delta > 0:
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

    def getBorder(self):
        points = np.array([self.step], dtype=np.int32)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        vetuex1 = [0,0]
        vetuex2 = [0,0]
        for vetuex in box:
            if vetuex[1]>vetuex1[1]:
                vetuex2 = vetuex1
                vetuex1 = vetuex
                
            elif vetuex[1]>vetuex2[1]:
                vetuex2 = vetuex
        
        min_y1 = 1000
        min_y2 = 1000

        for i in range(len(self.step)):
            if abs((self.step[i][0]-vetuex1[0])/(self.step[i][1]-vetuex1[1]+1e-10))<1 and self.step[i][1]<min_y1:
                min_y1 = self.step[i][1]
                point1 = self.step[i]
            if abs((self.step[i][0]-vetuex2[0])/(self.step[i][1]-vetuex2[1]+1e-10))<1 and self.step[i][1]<min_y2:
                min_y2 = self.step[i][1]
                point2 = self.step[i]
        
        print(point1, point2)
        # 求直线方程
        return self.getLine(point1, point2)

    def regionJudge(self, pose_point, score_threshold, box):
        """判断是否是上扶梯远处的点
            pose_point:目标点
            score_threshold:关键点置信度
            box:检测框
            return:
                True or False
        """
        if self.mode == 'normal':
            return True
        if pose_point[2]<score_threshold:
            return False

        distance = (self.A*pose_point[0]+self.B*pose_point[1]+self.C)/math.sqrt(self.A*self.A+self.B*self.B+1e-10)
        sign = self.C
        if distance * sign > 0 or self.is_in_poly(pose_point, box) == False:
            return False
        else:
            return True
            
    def is_in_poly(self, p, poly):
        """
    :param p: [x, y]
    :param poly: [[], [], [], [], ...]
    :return:
    """
        px, py, _ = p
        is_in = False
        for i, corner in enumerate(poly):
            next_i = i + 1 if i + 1 < len(poly) else 0
            x1, y1 = corner
            x2, y2 = poly[next_i]
            if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # if point is on vertex
                is_in = True
                break
            if min(y1, y2) < py <= max(y1, y2):  # find horizontal edges of polygon
                x = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-10)
                if x == px:  # if point is on edge
                    is_in = True
                    break
                elif x > px:  # if point is on left-side of line
                    is_in = not is_in
        return is_in

    def need_judge(self, pose_results, kpt_thr):
        """判断是否是待识别对象
            pose_results：17个人体姿态点
            kpt_thr：姿态点置信阈值

            return:
                flag: 是否是待识别对象
        """
        if self.is_in_poly(pose_results[15], self.larger_floor) or self.is_in_poly(pose_results[16], self.larger_floor):
            return False
        elif self.is_in_poly(pose_results[15], self.step) or self.is_in_poly(pose_results[16], self.step):
            return True
        else:
            points1 = self.getCrossPoints(self.kx, pose_results[11])
            if len(points1)>1 and (points1[0][0]-pose_results[11][0])*(points1[-1][0]-pose_results[11][0]) < 0:
                return True
            points2 = self.getCrossPoints(self.kx, pose_results[12])
            if len(points2)>1 and (points2[0][0]-pose_results[12][0])*(points2[-1][0]-pose_results[12][0]) < 0:
                return True
            return False

    def judge3DInvade(self, output, kpt_thr, vis_frame=None, error_thr=0.15):
        '''判断是否发生3D入侵

            output: 网络前向输出

            kpt_thr: 姿态点置信阈值

            error_thr: 容错阈值

            vis_frame: 输入视频,默认是release版本

            return:
                flag: 是否产生3D入侵警告
        '''

        assert self.kx!=None

        pose = output[6:]
        pose = np.reshape(pose,(17, 3))
        pose_score = output[4]
        box = [[output[:4][0],output[:4][1]], [output[:4][2],output[:4][1]], [output[:4][2],output[:4][3]], [output[:4][0],output[:4][3]]]

        flag = True
        
        for p in pose:
            flag = bool(p[2] > kpt_thr)
            if flag == False:
                break

        if pose_score < 0.5:#0.8
            return False

        if self.need_judge(pose, kpt_thr) == False:
            return False

        # cv2.circle(vis_frame, (776, 416), 5, (0, 255,0), 8)
        # cv2.circle(vis_frame, (613, 415), 5, (0, 255,0), 8)
        # 踢腿出界
        if self.judge2DborderIn(P=pose[15], score_threshold=kpt_thr, kx=self.kx) == False:
            if vis_frame is not None:
                cv2.circle(vis_frame, (int(pose[15][0]), int(pose[15][1])), 5, (0, 0, 255), 8)
                print('warning5: leg out of border')
            return True
        if self.judge2DborderIn(P=pose[16], score_threshold=kpt_thr, kx=self.kx) == False:
            if vis_frame is not None:
                cv2.circle(vis_frame, (int(pose[16][0]), int(pose[16][1])), 5, (0, 0, 255), 8)
                print('warning5: leg out of border')
            return True
        
         # 头是否歪了
        Crookedhead = False
        if pose[0][2] > kpt_thr and \
            pose[11][2] > kpt_thr and \
                pose[12][2] > kpt_thr:
            hip = (pose[11] + pose[12])/2
            if abs((pose[0][0]-hip[0])/(pose[0][1]-hip[1]+1e-10)) > 0.5:
                Crookedhead = True

        # 不是完整姿态
        if flag is False:
            in_border = 0
            out_border = 0
            for p in pose:
                if self.judge2DborderIn(P=p, score_threshold=kpt_thr, kx=self.kx): 
                    in_border += 1
                elif not self.judge2DborderIn(P=p, score_threshold=kpt_thr, kx=self.kx):
                    out_border += 1

            # 全部出界
            if in_border == 0 and out_border != 0 and Crookedhead is True:
                if vis_frame is not None:
                    cv2.circle(vis_frame, (int(pose[0][0]), int(pose[0][1])), 5, (0, 0, 255), 8)
                    print('warning1: out of border')
                return True
            # 内外都有，手在内部，头部歪曲
            elif self.judge2DborderIn(P=pose[9], score_threshold=kpt_thr, kx=self.kx, error_thr=error_thr) == True and \
                self.judge2DborderIn(P=pose[10], score_threshold=kpt_thr, kx=self.kx, error_thr=error_thr) == True and \
                    Crookedhead is True and out_border != 0:
                if vis_frame is not None:
                    cv2.circle(vis_frame, (int(pose[0][0]), int(pose[0][1])), 5, (0, 0, 255), 8)
                    print('warning2: out of border')
                return True
            # 内外都有，手部伸展
            elif out_border != 0 and in_border != 0:
                if self.judge2DborderIn(P=pose[9], score_threshold=kpt_thr, kx=self.kx, error_thr=0) == False:
                    if abs((pose[9][1]-pose[7][1])/abs(pose[9][0]-pose[7][0]+1e-10)) < 1 and abs((pose[5][1]-pose[7][1])/abs(pose[5][0]-pose[7][0]+1e-10)) < 2:
                        if vis_frame is not None:
                            cv2.circle(vis_frame, (int(pose[9][0]), int(pose[9][1])), 5, (0, 0, 255), 8)
                            print('warning3: out of border')
                        return True
                if self.judge2DborderIn(P=pose[10], score_threshold=kpt_thr, kx=self.kx, error_thr=0) == False:
                    if abs((pose[10][1]-pose[8][1])/abs(pose[10][0]-pose[8][0]+1e-10)) < 1 and abs((pose[6][1]-pose[8][1])/abs(pose[6][0]-pose[8][0]+1e-10)) < 2:
                        if vis_frame is not None:
                            cv2.circle(vis_frame, (int(pose[10][0]), int(pose[10][1])), 5, (0, 0, 255), 8)
                            print('warning4: out of border')
                        return True
                # if self.judge2DborderIn(P=pose[9], score_threshold=kpt_thr, kx=self.kx) == False:
                #     if vis_frame is not None:
                #         cv2.circle(vis_frame, (int(pose[9][0]), int(pose[9][1])), 5, (0, 0, 255), 8)
                #         print('warning7: out of border')
                #     return True
                # elif self.judge2DborderIn(P=pose[10], score_threshold=kpt_thr, kx=self.kx) == False:
                #     if vis_frame is not None:
                #         cv2.circle(vis_frame, (int(pose[10][0]), int(pose[10][1])), 5, (0, 0, 255), 8)
                #         print('warning8: out of border')
                #     return True
                if self.judge2DborderIn(P=pose[7], score_threshold=kpt_thr, kx=self.kx, error_thr=error_thr) == False:
                    if vis_frame is not None:
                        print(pose[7][0])
                        cv2.circle(vis_frame, (int(pose[7][0]), int(pose[7][1])), 5, (0, 0, 255), 8)
                        print('warning8: out of border')
                    return True
                if self.judge2DborderIn(P=pose[8], score_threshold=kpt_thr, kx=self.kx, error_thr=error_thr) == False:
                    if vis_frame is not None:
                        print(pose[8][0])
                        cv2.circle(vis_frame, (int(pose[8][0]), int(pose[8][1])), 5, (0, 0, 255), 8)
                        print('warning8: out of border')
                    return True
            return False

        # 完整姿态
        region_need_to_judge = self.regionJudge(pose[15], score_threshold=kpt_thr, box=box) and \
            self.regionJudge(pose[16], score_threshold=kpt_thr, box=box)

        if region_need_to_judge == False:
            cv2.circle(vis_frame, (36,36), 5, (0, 255, 0), 8)
            if abs((pose[9][1]-pose[7][1])/abs(pose[9][0]-pose[7][0]+1e-10)) < 1 \
                    and abs((pose[5][1]-pose[7][1])/abs(pose[5][0]-pose[7][0]+1e-10)) < 1 \
                        and self.judge2DfarBorderIn(pose[9], score_threshold=kpt_thr, error_thr=error_thr) \
                            and (pose[9][0]-pose[7][0])*(pose[5][0]-pose[7][0])<0:
                if vis_frame is not None:
                    cv2.circle(vis_frame, (int(pose[9][0]), int(pose[9][1])), 5, (0, 0, 255), 8)
                    print('warning6: out of border')
                return True
            elif abs((pose[10][1]-pose[8][1])/abs(pose[10][0]-pose[8][0]+1e-10)) < 1 \
                     and abs((pose[6][1]-pose[8][1])/abs(pose[6][0]-pose[8][0]+1e-10)) < 1 \
                         and self.judge2DfarBorderIn(pose[10], score_threshold=kpt_thr, error_thr=error_thr) \
                            and (pose[10][0]-pose[8][0])*(pose[6][0]-pose[8][0])<0:
                if vis_frame is not None:
                    cv2.circle(vis_frame, (int(pose[10][0]), int(pose[10][1])), 5, (0, 0, 255), 8)
                    print('warning6: out of border')
                return True
            elif Crookedhead == True  and self.judge2DfarBorderIn(pose[0], score_threshold=kpt_thr, error_thr=error_thr):
                if vis_frame is not None:
                    cv2.circle(vis_frame, (int(pose[0][0]), int(pose[0][1])), 5, (0, 0, 255), 8)
                    print('warning6: out of border')
                return True
            return False

        nose = pose[0]
        # # 伸头出界
        # if Crookedhead == True and self.judge2DborderIn(P=nose, score_threshold=kpt_thr, kx=self.kx, error_thr=0) == False:
        #     if vis_frame is not None:
        #         cv2.circle(vis_frame, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), 8)
        #         print('warning: out of border')
        #     return True
        

        shoulder = (pose[5] + pose[6]) / 2
        hip = (pose[11] + pose[12]) / 2
        ankle = (pose[15] + pose[16]) / 2
        knee = (pose[13]+pose[14]) / 2

        ky = (hip[0] - shoulder[0])/(hip[1] - shoulder[1]+1e-10) if shoulder[2]>ankle[2] else (hip[0] - ankle[0])/(hip[1] - ankle[1]+1e-10)

        nose_to_hip = np.array((nose[0]-hip[0], nose[1]-hip[1]))
        l_nose_to_hip=np.sqrt(nose_to_hip.dot(nose_to_hip))
        knee_to_hip = np.array((knee[0]-hip[0], knee[1]-hip[1]))
        l_knee_to_hip=np.sqrt(knee_to_hip.dot(knee_to_hip))
        cos_theta = nose_to_hip.dot(knee_to_hip)/(l_nose_to_hip*l_knee_to_hip)
        if hip[1] < nose[1] or cos_theta>math.cos(math.pi*5/6):
            ky = -self.kx

        # ky = -self.kx

        # 水平交点
        if pose[15][1] < pose[16][1]:
            LeftPoints = self.getLeftPoint(kx=self.kx, P=pose[16])
            RightPoints = self.getRightPoint(kx=self.kx, P=pose[16])
        else:
            LeftPoints = self.getLeftPoint(kx=self.kx, P=pose[15])
            RightPoints = self.getRightPoint(kx=self.kx, P=pose[15])
        
        # 得到最近的两个水平交点
        if len(LeftPoints)!=0 and len(RightPoints)!=0:
            point1 = LeftPoints[-1]
            point2 = RightPoints[0]
            # crossPoints = self.getRightPoint(kx=self.kx, P=pose[15])

        # 
        # if len(crossPoints) > 3:
        #     if pose[15][1] < pose[16][1]:
        #         point1, point2 = self.getNearestCrossPoints(crossPoints, pose[16])
        #     else:
        #         point1, point2 = self.getNearestCrossPoints(crossPoints, pose[15])

            if vis_frame is not None:
                cv2.circle(vis_frame, tuple(point1), 5, (255, 0, 0), 8)
                cv2.circle(vis_frame, tuple(point2), 5, (255, 0, 0), 8)
            
            tolerable_eer_thr = abs(point1[0] - point2[0])*error_thr


            # pose[2:11, :], pose[0:2, :] = pose[0:9, :].copy(), pose[9:11, :].copy()
            pose[3:11, :], pose[1:3, :] = pose[1:9, :].copy(), pose[9:11, :].copy()

            # 伸头出界
            if Crookedhead == True and self.judge2DborderIn(P=nose, score_threshold=kpt_thr, kx=self.kx, error_thr=0) == False:
                p = pose[0]
                distance1 = self.getDist_P2L_V2(p, -1/(self.kx+1e-10), point1)
                distance2 = self.getDist_P2L_V2(p, -1/(self.kx+1e-10), point2)
                parallelLineDistance = self.getDist_P2L_V2(point1, -1/(self.kx+1e-10), point2)
                if abs(distance1+distance2-parallelLineDistance)/2 >= tolerable_eer_thr*0.1:
                    if vis_frame is not None:
                        cv2.circle(vis_frame, (int(nose[0]), int(nose[1])), 5, (0, 0, 255), 8)
                        print('warning: out of border')
                    return True

            # 伸手出界
            if (abs((pose[1][1]-pose[9][1])/abs(pose[1][0]-pose[9][0]+1e-10)) < 1 and abs((pose[7][1]-pose[9][1])/abs(pose[7][0]-pose[9][0]+1e-10)) < 2) or \
                (abs((pose[2][1]-pose[10][1])/abs(pose[2][0]-pose[10][0]+1e-10)) < 1 and abs((pose[8][1]-pose[10][1])/abs(pose[8][0]-pose[10][0]+1e-10)) < 2):
            # if Crookedhead == True and self.judge2DborderIn(P=nose, score_threshold=kpt_thr, kx=self.kx) == True:
                for i in range(1,3):
                    p = pose[i]
                    distance1 = self.getDist_P2L_V2(p, -1/(self.kx+1e-10), point1)
                    distance2 = self.getDist_P2L_V2(p, -1/(self.kx+1e-10), point2)
                    parallelLineDistance = self.getDist_P2L_V2(point1, -1/(self.kx+1e-10), point2)

                    # if abs(distance1+distance2-parallelLineDistance)/2 >= tolerable_eer_thr:
                    if abs(distance1+distance2-parallelLineDistance)/2 >= tolerable_eer_thr:
                        if vis_frame is not None:
                            # print(point1, point2, p)
                            # print(-self.kx, distance1, distance2, parallelLineDistance, abs(distance1+distance2-parallelLineDistance)/2, tolerable_eer_thr)
                            print('warning: hand out of border')
                            cv2.circle(vis_frame, (int(p[0]), int(p[1])), 10, (0, 0, 255), 8)
                        return True
                ky = -self.kx
                # tolerable_eer_thr = abs(point1[0]-point2[0])*0.125 
                # tolerable_eer_thr = tolerable_eer_thr

            parallelLineDistance = self.getDist_P2L_V2(point1, 1/(ky+1e-10), point2)
            for i in range(1, 17):
                p = pose[i]
                distance1 = self.getDist_P2L_V2(p, 1/(ky+1e-10), point1)
                distance2 = self.getDist_P2L_V2(p, 1/(ky+1e-10), point2)
                
                if abs(distance1+distance2-parallelLineDistance)/2 >= tolerable_eer_thr:
                    if vis_frame is not None:
                        # print(abs((pose[0][0]-hip[0])/(pose[0][1]-hip[1]+1e-10)))
                        # print(abs(distance1+distance2-parallelLineDistance)/2, tolerable_eer_thr)
                        print('warning: out of border')
                        print(pose)
                        cv2.circle(vis_frame, (int(p[0]), int(p[1])), 10, (0, 0, 255), 8)
                    return True

        return False



if __name__ == "__main__":
    detector = Detector("E:\\alert\demo\demo1\demov21_.json")
    # detector = Detector("E:\\alert\demo\demo1\Seg1.json")
    pass
