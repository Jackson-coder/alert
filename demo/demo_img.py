import json

def getLine(a,b):
    """计算直线方程
        A*x+B*y+C=0
        a：直线a点坐标
        b：直线b点坐标
    """
    A=a[1]-b[1]
    B=b[0]-a[0]
    C=a[0]*b[1]-a[1]*b[0]

    return A,B,C

 
#***** 点到直线的距离:P到AB的距离*****
#P为线外一点，AB为线段两个端点
def getDist_P2L(P,a,b):
    """计算点到直线的距离
        P：定点坐标
        a：直线a点坐标
        b：直线b点坐标
    """
    #求直线方程
    A,B,C = getLine(a,b)
    #代入点到直线距离公式
    distance=0
    distance=(A*P[0]+B*P[1]+C)/math.sqrt(A*A+B*B)
    
    return distance

def getCrossPoint(k,P,a,b):
    """计算直线与直线的交点
        k: 直线1 斜率
        P：直线1 定点坐标
        a：直线2 a点坐标
        b：直线2 b点坐标
    """
    b0 = P[1] - k*P[0]
    A,B,C = getLine(a,b)
    x = -(B*b0+C)/(A+B*k)
    y = -k*(B*b0+C)/(A+B*k)+b0

    return [x,y]

def judgeCross(k,P,a,b):
    """判断直线与线段是否相交
        k: 直线1 斜率
        P：直线1 定点坐标
        a：线段2 a点坐标
        b：线段2 b点坐标
    """
    crossPoint = getCrossPoint(k,P,a,b)
    if (crossPoint[0]-a[0])*(crossPoint[0]-b[0])<=0 and (crossPoint[1]-a[1])*(crossPoint[1]-b[1])<=0:
        return crossPoint, True
    else:
        return crossPoint, False


with open('/home/lyh/mmpose/tests/data/coco/1.json','r',encoding='utf8')as fp:
    json_data = json.load(fp)
    alert1 = json_data['shapes'][0]['points']
    alert2 = json_data['shapes'][1]['points']
    P = [960,540]
    print(alert1)
    for i in range(len(alert1)):
        crossPoint, flag = judgeCross(0,P,alert1[i],alert1[(i+1)%len(alert1)])
        if flag:
            print(crossPoint)
    for i in range(len(alert2)):
        crossPoint, flag = judgeCross(0,P,alert2[i],alert2[(i+1)%len(alert2)])
        if flag:
            print(crossPoint)

    alert2 = json_data['shapes'][1]['points']
    print('这是文件中的json数据：',json_data['shapes'][0]['points'])
    print('这是读取到文件数据的数据类型：', type(json_data))