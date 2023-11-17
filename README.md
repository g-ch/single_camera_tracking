# single_camera_tracking
This package receives RGB images from a camera and does instance segmentation to find the objects in the image. Using the results of Superpoint and Superglue, this package tracks objects in the image and publishes the tracking ID and the masks of the objects and the Keypoint positions on the masks in the current frame and last frame. 

____
## Requirements Installation
__Tested environment: Ubuntu 20.04 + ROS Noetic__

This package uses [SuperPoint-SuperGlue-TensorRT](https://github.com/yuefanhao/SuperPoint-SuperGlue-TensorRT) and [mmdetection](https://github.com/open-mmlab/mmdetection). Therefore, you need to install the required environment for these two projects first.

### SuperPoint-SuperGlue-TensorRT
- [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive) and cudnn. CUDA 11.x should all works.
- [CUDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux-tar). Using tar file installation is suggested. Unpack and move the files to the folder given in the instructions.
- [TensorRT 8.4](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar). Using tar file installation is suggested. After you have unpacked TensorRT tar file, add "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxx/TensorRT-8.4.1.5/lib", where "xxx" is the installation path of TensorRT, to .bashrc.
- OpenCV 4. The version installed with ROS Noetic is sufficient.
- Eigen. The version installed with ROS Noetic is sufficient.
- [yaml-cpp](https://github.com/jbeder/yaml-cpp). Follow the instructions in the link, make and sudo make install.


### mmdetection
Refer to installation in [mmdetection](https://github.com/open-mmlab/mmdetection).

After install mmdetection, you need to download the pretrained model and config file. _TODO:Specifiy which one to download._

### ROS Python Package
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
- Run segmentation node by
```
conda activate openmmlab
rosrun single_camera_tracking instance_segmentation.py
```

- Run superpoint_superglue and tracking node by
```
conda activate openmmlab
rosrun single_camera_tracking tracking
``` 
Note the first time you run this node you will see ```Building inference engine......```. It will take about 15 minutes to 1 hour to build. 

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