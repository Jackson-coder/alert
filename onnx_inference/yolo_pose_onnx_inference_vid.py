'''
Description: 
Version: 0.1.1
Author: Jackson-coder
Date: 2022-08-06 13:13:44
LastEditors: Jackson-coder
LastEditTime: 2022-08-08 10:02:40
'''
import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm

from video_init import getHorizonSlope, getVerticalSlope
from video_detect import judge3DInvade


_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5

def read_img(img, img_mean=127.5, img_scale=1/127.5):
    img = (img - img_mean) * img_scale
    img = np.asarray(img, dtype=np.float32)
    img = np.expand_dims(img,0)
    img = img.transpose(0,3,1,2)
    return img


def model_inference(model_path=None, input=None):
    #onnx_model = onnx.load(args.model_path)
    session = onnxruntime.InferenceSession(model_path, None)
    input_name = session.get_inputs()[0].name
    output = session.run([], {input_name: input})
    return output


def model_inference_vid(model_path, kpt_thr=0.3, resize=960, scale=1, tolerable_eer_thr=25, kx=0.0, ky=None,  \
    make_border=True, save_out_video=True, dst_path=None, json_file=None, direction='forward', vid_path=None, \
        mean=0.0, sigma=0.00392156862745098, mode='video_inference'):
    '''判断是否发生3D入侵
        model_path: 模型文件路径

        kpt_thr： 姿态点置信阈值

        resize: 图片放缩后的大小，默认960x960，送入神经网络训练的输入特征图

        scale: 处理视频帧相对于原始视频帧的缩小倍数，默认为1（make_border=True时处理完的姿态点已映射到原图，若make_border=False需要手动调整）

        tolerable_eer_thr: 针对完整17关节点识别对象的边界容错阈值

        kx:水平线斜率
        
        ky:垂直线斜率

        make_border: 使用对原始图上下加边框后送入网络预测，建议为True

        save_out_video: 是否保存输出处理图

        dst_path: 输出视频路径

        json_file: 图像分割结果
                        
        direction:识别对象行进方向

        vis_path: 输入视频路径
        
        mean，sigma: 输入图像归一化因子

        mode: 运行模式, 'video_inference' 或 'video_init'

        return:
            vis_frame: 加入3D入侵警告的视频帧
    '''
        
    os.makedirs(dst_path, exist_ok=True)

    assert scale!=0

    cap = cv2.VideoCapture(vid_path)

    if save_out_video and make_border is False:
        fps = 20
        size = (resize, resize)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(dst_path,
                         f'vis_{os.path.basename(vid_path)}'), fourcc,
            fps, size)
    elif save_out_video and make_border is True:
        fps = 20
        size = (1280, 720)#原视频
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(dst_path,
                         f'vis_{os.path.basename(vid_path)}'), fourcc,
            fps, size)

    frame_id = 0
    ky_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        img = frame[:, :, ::-1]

        if mode == 'video_init'and frame_id == 0:
            kx = getHorizonSlope(frame)

        if make_border:  # (recommand)
            resize_coefficient = img.shape[1]/resize
            height, width = int(img.shape[0]/resize_coefficient), resize
            add_bottom_and_up = (width - height)//2
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            img = cv2.copyMakeBorder(img,add_bottom_and_up,add_bottom_and_up,0,0,cv2.BORDER_CONSTANT)

            input = read_img(img, mean, sigma)
            output = model_inference(model_path, input) 

            pose_results = output[0][:, 6:]
            pose_results = np.reshape(pose_results,(-1, 17, 3))

            pose_results[:,:,1] = pose_results[:,:,1] - add_bottom_and_up
            pose_results[:,:,:2] = pose_results[:,:,:2]*resize_coefficient
            output[0][:, 6:] = pose_results.reshape((-1,17*3))
            pose_scores = output[0][:, 4]

            img = frame
            
        else:
            img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
            input = read_img(img, mean, sigma)
            output = model_inference(model_path, input) #没有放缩回去，待完善
        
            pose_results = output[0][:, 6:]
            pose_results = np.reshape(pose_results,(-1, 17, 3))
            pose_scores = output[0][:, 4]

        vis_frame = post_process(img, output[0], score_threshold=0.3)

        if mode == 'video_init':
            ky_buffer.append(getVerticalSlope(pose_results, kpt_thr))

            if len(ky_buffer)<100:
                ky = np.mean(ky_buffer)
                break
        else:
            vis_frame = judge3DInvade(json_file, pose_results, pose_scores, direction, kpt_thr,
                                        vis_frame, tolerable_eer_thr, kx, ky, scale)

        if save_out_video:
            videoWriter.write(vis_frame)
            cv2.imwrite('test.jpg', vis_frame)

        frame_id+=1

    if mode == 'video_init':
        print('\n----------水平方向(角度)：',kx,'----------')
        print('------垂直方向（x/y）：',ky,'------')
    

    if save_out_video:
        videoWriter.release()
        print('release')

    cap.release()

"""
----------------output------------------

output[0:4]:  det_bbox[1], det_bbox[0], det_bbox[3], det_bbox[2]
output[4]: det_scores[idx]
output[5]: det_labels[idx]
output[6:]: kpts  (x,y,score)
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
"""
def post_process(img, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]  #det_labels[idx], det_scores[idx], det_bbox[1], det_bbox[0], det_bbox[3], det_bbox[2]

    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.
    for idx in range(len(det_bboxes)):
        det_bbox = det_bboxes[idx]
        kpt = kpts[idx]
        if det_scores[idx]>score_threshold:
            plot_skeleton_kpts(img, kpt)
        # plot_skeleton_kpts(img, kpt)
    return img


def plot_skeleton_kpts(im, kpts, steps=3):
    num_kpts = len(kpts) // steps
    #plot keypoints
    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        conf = kpts[steps * kid + 2]
        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    #plot skeleton
    for sk_id, sk in enumerate(skeleton):
        r, g, b = pose_limb_color[sk_id]
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)






def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="./yolov5l6_pose_960_ti_lite_54p9_82p2.onnx")
    parser.add_argument("--vid-path", type=str, default="../data/video/demo.mp4")
    parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
    parser.add_argument(
        '--kpt-thr', type=float, default=0.4, help='Keypoint score threshold')
    parser.add_argument(
        '--kx',
        type=float,
        default=0.0,
        help='Slope of horizontal line')
    parser.add_argument(
        '--ky',
        type=float,
        default=-0.02361783, #-0.09467121147881472,
        help='Slope of vertical line')
    parser.add_argument(
        '--resize',
        type=int,
        default=960,
        help='Input size of yolopose-net')
    parser.add_argument(
        '--scale',
        type=float,
        default=1,
        help='Ratio of sizes of Original frame and frame with Keypoints')
    parser.add_argument(
        '--tolerable-eer-thr', type=float, default=25, help='Keypoint score threshold')
    parser.add_argument(
        '--json-file',
        type=str,
        default='./1.json',
        help='image segmentation result file of alert')
    parser.add_argument(
        '--direction',
        type=str,
        default='forward',
        help='People direction to detect,\'forward\'or \'backward\'. Default: \'forward\'')
    parser.add_argument(
        '--mode',
        type=str,
        default='video_inference',
        help='inference mode, \'video_init\' or \'video_inference\' ')
    parser.add_argument(
        '--make-border',
        action="store_false",
        help='Whether to use the original graph to add a border up and down to feed into the network prediction')
    parser.add_argument(
        '--save-out-video',
        action="store_false",
        help='Save the video or not')

    args = parser.parse_args()

    model_inference_vid(model_path=args.model_path, kpt_thr=args.kpt_thr, resize=args.resize, 
                               kx=args.kx, ky=args.ky, vid_path=args.vid_path,
                               tolerable_eer_thr=args.tolerable_eer_thr, scale=args.scale,
                               json_file=args.json_file,#'/home/lyh/edgeai-yolov5/onnx_inference/1.json',
                               direction=args.direction, make_border=args.make_border, save_out_video=args.save_out_video, 
                               dst_path=args.dst_path, mode=args.mode, mean=0.0, sigma=0.00392156862745098)#, mode='video_init'


if __name__== "__main__":
    main()
