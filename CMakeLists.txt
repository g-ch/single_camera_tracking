cmake_minimum_required(VERSION 3.5)
project(single_camera_tracking)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "release")
add_definitions(-w)

add_subdirectory(${PROJECT_SOURCE_DIR}/SuperPoint-SuperGlue-TensorRT/3rdparty/tensorrtbuffer)

add_definitions(${PCL_DEFINITIONS})

find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  mask_kpts_msgs
)

find_package(OpenCV 4.2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)


catkin_package()

# change this to your TensorRT root directory
set(TENSORRT_ROOT $ENV{HOME}/Documents/TensorRT-8.4.1.5) 

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/SuperPoint-SuperGlue-TensorRT
  ${PROJECT_SOURCE_DIR}/SuperPoint-SuperGlue-TensorRT/include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${TENSORRT_ROOT}/include
)

add_library(${PROJECT_NAME}_superglue_lib SHARED
  ${PROJECT_SOURCE_DIR}/SuperPoint-SuperGlue-TensorRT/src/super_point.cpp
  ${PROJECT_SOURCE_DIR}/SuperPoint-SuperGlue-TensorRT/src/super_glue.cpp
)

target_link_libraries(${PROJECT_NAME}_superglue_lib
#   nvinfer
  ${TENSORRT_ROOT}/lib/libnvinfer.so  
#   nvonnxparser
  ${TENSORRT_ROOT}/lib/libnvonnxparser.so 
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  yaml-cpp
  tensorrtbuffer
)


add_executable(tracking src/tracking.cpp)
target_link_libraries(tracking ${PROJECT_NAME}_superglue_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})

add_executable(superpoint_test src/superpoint_test.cpp)
target_link_libraries(superpoint_test ${PROJECT_NAME}_superglue_lib ${PCL_LIBRARIES} ${catkin_LIBRARIES} ${OpenCV_LIBS})
