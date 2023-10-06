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

After install mmdetection, you need to download the pretrained model and config file. _TODO:Specifiy which one to download._

### Ours
In the step to install dependencies for mmdetection, you should have created an annaconda virtual environment (default name: openmmlab). Then you need to install rospkg in this environment.
```
conda activate openmmlab
pip install -U rospkg
```

___
## Compile
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

___
## Run


___
## Trouble Shooting
#### 1. Libffi Version Problem
When using anaconda to run mmdetection in ```instance_segmentation.py```. There might be a version mismatching problem of libffi. You will see
```
ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
```
To solve this issue, refer to [this](https://aitechtogether.com/python/92590.html). For non-Chinese readers, we translate the procedures in the following.

Open the lib folder of your anaconda or miniconda environment installation folder. For example, ```cd /home/anaconda3/envs/xxx/lib```. Excute `ls -l` and you will see `libffi.so.7` is linked to `libffi.so.8.1.0`. 

Backup `libffi.so.7` and relink it to `libffi.so.7.1.0` using the following commands.
```
mv libffi.so.7 libffi.so.7.backup
sudo ln -s /lib/x86_64-linux-gnu/libffi.so.7.1.0 libffi.so.7
sudo ldconfig
```
The problem should be solved.