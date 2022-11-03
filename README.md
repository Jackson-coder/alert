<!--
 * @Description: 
 * @Version: 
 * @Author: Jackson-coder
 * @Date: 2022-08-07 19:46:17
 * @LastEditors: Jackson-coder
 * @LastEditTime: 2022-11-03 10:43:08
-->
# YOLO-Pose Multi-person Pose estimation model


Given below is a sample with Yolo-Pose on a crowded sample.
**At the end of the article, the principle of the three-dimensional intrusion algorithm is shown.**

<br/> 
<p float="left">
<img width="640" src="./result/demo.gif">
</p>   



## **Model Inference**


###  **ONNX Export Including Detection and Pose Estimation:**
* Run the following command to export the entire models including the detection part, 
    ``` 
    python models/export.py --weights weights/yolov5l6_pose_832_mix-finetune.pt --img 960 --batch 1 --simplify --export-nms
    ```
* Apart from exporting the complete ONNX model, above script will generate a prototxt file that contains information of the detection layer. This prototxt file is required to deploy the moodel on TI SoC.

###  **ONNXRT Inference: Human Pose Estimation Inference with an End-to-End ONNX Model:**

 * If you haven't exported a model with the above command, download a sample model from this [link](http://software-dl.ti.com/jacinto7/esd/modelzoo/gplv3/latest/edgeai-yolov5/pretrained_models/models/keypoint/coco/edgeai-yolov5/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx).
 * Run the script as below to run inference with an ONNX model. The script runs inference and visualize the results. There is no extra post-processing required. The ONNX model is self-sufficient unlike existing bottom-up approaches. The [script](onnx_inference/yolo_pose_onnx_inference.py) is compleletey independent and contains all perprocessing and visualization. 
    ``` 
    cd onnx_inference
    python yolo_pose_onnx_inference.py --model-path ../weights/yolov5l6_pose_832_mix-finetune.onnx --img-path ../img_data.txt --dst-path ../result
    ```
    
    ``` 
    cd onnx_inference
    python yolo_pose_onnx_inference_vid.py --model-path ../weights/yolov5l6_pose_832_mix-finetune.onnx --vid-path ../data/video/demo.mp4 --dst-path ../result
    ```
    
###  **Run by just a simple pt weight file**
```
python detect.py --device 0 --weights weights/yolov5l6_pose_832_mix-finetune.pt --kpt-label --view-img
```

##  **Algorithm Principle**
Click [script](Algorithm_Principle.docx) for details.

## **References**

[1] [Official YOLOV5 repository](https://github.com/ultralytics/yolov5/) <br>
[2] [Official YOLO-Pose repository](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose) <br>
[3] [yolov5-improvements-and-evaluation, Roboflow](https://blog.roboflow.com/yolov5-improvements-and-evaluation/) <br>
[4] [Focus layer in YOLOV5]( https://github.com/ultralytics/yolov5/discussions/3181) <br>
[5] [CrossStagePartial Network](https://github.com/WongKinYiu/CrossStagePartialNetworkss)  <br>
[6] Chien-Yao Wang, Hong-Yuan Mark Liao, Yueh-Hua Wu, Ping-Yang Chen, Jun-Wei Hsieh, and I-Hau Yeh. [CSPNet: A new backbone that can enhance learning capability of
cnn](https://arxiv.org/abs/1911.11929). Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshop (CVPR Workshop),2020. <br>
[7]Shu Liu, Lu Qi, Haifang Qin, Jianping Shi, and Jiaya Jia. [Path aggregation network for instance segmentation](https://arxiv.org/abs/1803.01534). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 8759–8768, 2018 <br>
[8] [Efficientnet-lite quantization](https://blog.tensorflow.org/2020/03/higher-accuracy-on-vision-models-with-efficientnet-lite.html) <br>
[9] [YOLOv5 Training video from Texas Instruments](https://training.ti.com/process-efficient-object-detection-using-yolov5-and-tda4x-processors) <br> 
[10] [YOLO-Pose Training video from Texas Instruments:Upcoming](Upcoming)