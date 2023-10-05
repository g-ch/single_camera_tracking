# single_camera_tracking
This package receives RGB images from a camera and does instance segmentation to find the objects in the image. Using the results of Superpoint and Superglue, this package tracks objects in the image and publishes the tracking ID and the masks of the objects and the Keypoint positions on the masks in the current frame and last frame. 

____
## Requirements Installation
__Tested environment: Ubuntu 20.04 + ROS Noetic__

This package uses [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT) and [mmdetection](https://github.com/open-mmlab/mmdetection). Therefore, you need to install the required environment for these two projects first.

### SuperPoint-SuperGlue-TensorRT
- [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive). CUDA 11.x should all works.
- [TensorRT 8.4](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar). Use tar file installation.
- OpenCV 4
- Eigen
- yaml-cpp

### mmdetection
Refer to installation in [mmdetection](https://github.com/open-mmlab/mmdetection).

### Ours

```

```

___
# Compile
```
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone --recursive git@github.com:g-ch/single_camera_tracking.git
```

Before you run ```catkin build```, you need to change ```set(TENSORRT_ROOT xxx) ``` in CMakeList.txt to the installation directory of your TensorRT. And add the following lines in SuperPoint-SuperGlue-TensorRT/3rdparty/tensorrtbuffer/CmakeList.txt
```
set(TENSORRT_ROOT xxx) 
include_directories(
	${TENSORRT_ROOT}/include
)
```
Replace ```xxx``` with your TensorRT installation directory.

Then run ```catkin build``` in ```catkin_ws``` directory.
