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

    def getHorizonSlope(self, cur_frame):
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

    def getVerticalSlope(self, pose_results, kpt_thr):

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
        x = -(B*b0+C)/(A+B*k)
        y = -k*(B*b0+C)/(A+B*k)+b0

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

    def cfg_init(self, output, frame, kpt_thr):

        pose_results = output[0][:, 6:]
        pose_results = np.reshape(pose_results,(-1, 17, 3))
        pose_scores = output[0][:, 4]

        if self.kx == None:
            self.kx = self.getHorizonSlope(frame)

        self.__ky_buffer.append(self.getVerticalSlope(pose_results, kpt_thr))

        if len(self.__ky_buffer)>100:
            self.ky = np.mean(self.__ky_buffer)

        if self.kx != None and self.ky != None:
            print('\n----------水平方向(角度)：',self.kx,'----------')
            print('------垂直方向（x/y）：',self.ky,'------')

    def judge3DInvade(self, output, json_file, direction, kpt_thr, vis_frame, tolerable_eer_thr, scale=1):
        '''判断是否发生3D入侵
            json_file: 图像分割结果

            direction:识别对象行进方向

            kpt_thr: 姿态点置信阈值

            vis_frame: 输入视频

            tolerable_eer_thr: 针对完整17关节点识别对象的边界容错阈值
            
            scale: 处理输出的视频帧相对于原始视频帧的缩小倍数

            return:
                vis_frame: 加入3D入侵警告的视频帧
        '''

        assert self.kx!=None and self.ky!=None

        pose_results = output[0][:, 6:]
        pose_results = np.reshape(pose_results,(-1, 17, 3))
        pose_scores = output[0][:, 4]

        for pose, pose_score in zip(pose_results, pose_scores):
            flag = True
            for p in pose:
                flag = bool(p[2] > kpt_thr)
                if flag is False:
                    break

            # 只识别面朝前或者面朝后的人
            if direction != self.judgeDirection(pose, kpt_thr): # judgeDirection 将在第三版代码更新后去除，将使用跟踪位移方向判断人的移动方向
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
                    if self.judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=self.kx, scale=scale): 
                        in_border += 1
                    elif not self.judge2DborderIn(json_file, P=p, score_threshold=kpt_thr, kx=self.kx, scale=scale):
                        out_border += 1

                # 全部出界
                if in_border == 0 and out_border != 0 and Crookedhead is True:
                    cv2.circle(vis_frame, [int(pose[0][0]),
                            int(pose[0][1])], 5, (0, 0, 255), 8)
                    print('warning1: out of border')
                # 内外都有，手在内部，头部歪曲
                elif self.judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=self.kx, scale=scale) == True and \
                    self.judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=self.kx, scale=scale) == True and \
                        Crookedhead is True and out_border != 0 and in_border != 0:
                    cv2.circle(vis_frame, [int(pose[0][0]),
                            int(pose[0][1])], 5, (0, 0, 255), 8)
                    print('warning2: out of border')
                # 内外都有，手部伸展
                elif out_border != 0 and in_border != 0:
                    if self.judge2DborderIn(json_file, P=pose[9], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
                        if abs((pose[9][1]-pose[7][1])/(pose[9][0]-pose[7][0])) < 2:
                            p = [int(pose[9][0]), int(pose[9][1])]
                            cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                            print('warning3: out of border')
                    elif self.judge2DborderIn(json_file, P=pose[10], score_threshold=kpt_thr, kx=self.kx, scale=scale) == False:
                        if abs((pose[10][1]-pose[8][1])/(pose[10][0]-pose[8][0])) < 2:
                            p = [int(pose[10][0]), int(pose[10][1])]
                            cv2.circle(vis_frame, p, 5, (0, 0, 255), 8)
                            print('warning4: out of border')

                continue

            # 完整姿态
            # 四个水平交点
            crossPoints = self.getCrossPoints(P=pose[15])

            if len(crossPoints) != 0:
                # print(crossPoints)
                for point in crossPoints:
                    cv2.circle(vis_frame, point, 5, (255, 0, 0), 8)
                parallelLineDistance = self.getDist_P2L_V2(
                    crossPoints[1], 1/self.ky, crossPoints[2])
                for p in pose:
                    distance1 = self.getDist_P2L_V2(p, 1/self.ky, crossPoints[1])
                    distance2 = self.getDist_P2L_V2(p, 1/self.ky, crossPoints[2])
                    if abs(distance1+distance2-parallelLineDistance) > tolerable_eer_thr:
                        # print(abs(distance1+distance2-parallelLineDistance))
                        print('warning: out of border')
                        cv2.circle(vis_frame, [int(pose[0][0]), int(
                            pose[0][1])], 10, (0, 0, 255), 8)

        return vis_frame




