<div align="center">
  <img src="resources/mmpose-logo.png" width="450"/>
  <div>&nbsp;</div>
  <div align="center">
    <b>OpenMMLab website</b>
    <sup>
      <a href="https://openmmlab.com">
        <i>HOT</i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b>OpenMMLab platform</b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i>TRY IT OUT</i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![Documentation](https://readthedocs.org/projects/mmpose/badge/?version=latest)](https://mmpose.readthedocs.io/en/latest/?badge=latest)
[![actions](https://github.com/open-mmlab/mmpose/workflows/build/badge.svg)](https://github.com/open-mmlab/mmpose/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmpose/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmpose)
[![PyPI](https://img.shields.io/pypi/v/mmpose)](https://pypi.org/project/mmpose/)
[![LICENSE](https://img.shields.io/github/license/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/blob/master/LICENSE)
[![Average time to resolve an issue](https://isitmaintained.com/badge/resolution/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)
[![Percentage of issues still open](https://isitmaintained.com/badge/open/open-mmlab/mmpose.svg)](https://github.com/open-mmlab/mmpose/issues)

[📘Documentation](https://mmpose.readthedocs.io/en/v0.28.0/) |
[🛠️Installation](https://mmpose.readthedocs.io/en/v0.28.0/install.html) |
[👀Model Zoo](https://mmpose.readthedocs.io/en/v0.28.0/modelzoo.html) |
[📜Papers](https://mmpose.readthedocs.io/en/v0.28.0/papers/algorithms.html) |
[🆕Update News](https://mmpose.readthedocs.io/en/v0.28.0/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmpose/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_CN.md)

</div>

## Introduction

MMPose is an open-source toolbox for pose estimation based on PyTorch.
It is a part of the [OpenMMLab project](https://github.com/open-mmlab).

The master branch works with **PyTorch 1.5+**.

<!-- <iframe height=640 width=640 src="vis_results/vis_demo_v.mp4"> -->

![image](./vis_results/hrnet.gif)


## Installation

MMPose depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.
Please refer to [install.md](docs/en/install.md) for detailed installation guide.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
pip3 install -e .
```

## Getting Started

### 1.Video Parameter Initial Config

```shell
CUDA_VISIBLE_DEVICES=0 python demo/video_cfg_init.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --video-path demo/resources/demo_v.mp4
```

### 2.3DInvade Judging
```shell
CUDA_VISIBLE_DEVICES=0 python demo/top_down_video_demo_with_mmdet_alert_v2.py \
    demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
    https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth \
    configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py \
    https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth \
    --video-path demo/resources/demo_v.mp4 \
    --out-video-root vis_results
```


## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```
