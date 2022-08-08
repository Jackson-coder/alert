<!--
 * @Description: 
 * @Version: 
 * @Author: Jackson-coder
 * @Date: 2022-08-07 19:46:17
 * @LastEditors: Jackson-coder
 * @LastEditTime: 2022-08-08 11:17:43
-->
# YOLO-Pose Multi-person Pose estimation model


Given below is a sample comparision with existing Associative Embedding based approach with HigherHRNet on a crowded sample.
**On the left is Output from HigherHRNetW32 and on the right is Output from YOLOv5l6-pose.**

<br/> 
<p float="left">
<img width="1000" src="./result/cmp.gif">
</p>   



## **Model Inference**

<br/> 

###  **ONNX Export Including Detection and Pose Estimation:**
* Run the following command to export the entire models including the detection part, 
    ``` 
    python models/export.py --weights weights/yolopose/yolov5m6_pose_960_ti_lite.pt --img 960 --batch 1 --simplify --export-nms
    ```
* Apart from exporting the complete ONNX model, above script will generate a prototxt file that contains information of the detection layer. This prototxt file is required to deploy the moodel on TI SoC.

###  **ONNXRT Inference: Human Pose Estimation Inference with an End-to-End ONNX Model:**

 * If you haven't exported a model with the above command, download a sample model from this [link](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/latest/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx).
 * Run the script as below to run inference with an ONNX model. The script runs inference and visualize the results. There is no extra post-processing required. The ONNX model is self-sufficient unlike existing bottom-up approaches. The [script](onnx_inference/yolo_pose_onnx_inference.py) is compleletey independent and contains all perprocessing and visualization. 
    ``` 
    cd onnx_inference
    python yolo_pose_onnx_inference.py --model-path ../weights/yolopose/yolov5m6_pose_960_ti_lite.onnx --img-path ../img_data.txt --dst-path ../result
    ```
    
    ``` 
    cd onnx_inference
    python yolo_pose_onnx_inference_vid.py --model-path ../weights/yolopose/yolov5m6_pose_960_ti_lite.onnx --vid-path ../data/video/demo.mp4 --dst-path ../result
    ```
    

## **References**

[1] [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/) <br>
[2] [yolov5-improvements-and-evaluation, Roboflow](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) <br>
[3] [Focus layer in YOLOV5]( https://github.com/ultralytics/yolov5/discussions/3181) <br>
[4] [CrossStagePartial Network](https://github.com/WongKinYiu/CrossStagePartialNetworkss)  <br>
[5] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. [CSPNet: A new backbone that can enhance learning capability of
cnn](https://arxiv.org/abs/1911.11929). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPR Workshop),2020. <br>
[6]Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. [Path aggregation network for instance segmentation](https://arxiv.org/abs/1803.01534). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8759–8768, 2018 <br>
[7] [Efficientnet-lite quantization](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) <br>
[8] [YOLOv5 Training video from Texas Instruments](https://training.ti.com/process-efficient-object-detection-using-yolov5-and-tda4x-processors) <br> 
[9] [YOLO-Pose Training video from Texas Instruments:Upcoming](Upcoming)