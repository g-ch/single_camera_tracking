#!/usr/bin/env python
#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import os
import fnmatch
import argparse
import rospy
import tf.transformations as tfm
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

        
init_pose_recorded = False
        

def generate_point_cloud(depth_image, K):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    rows, cols = depth_image.shape
    x = np.linspace(0, cols-1, cols)
    y = np.linspace(0, rows-1, rows)
    x, y = np.meshgrid(x, y)

    x = (x - cx) * depth_image / fx
    y = (y - cy) * depth_image / fy
    z = depth_image

    point_cloud = np.stack([x, y, z], axis=-1)
    points = point_cloud.reshape(-1, 3)

    return points


def visualize_point_cloud(points, point_size=1.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()


def scan_files_with_ext(dir, ext):
    files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if fnmatch.fnmatch(file, ext):
                files.append(os.path.join(root, file))

    return files        

def read_pose_txt(pose_txt):
    with open(pose_txt, 'r') as f:
        lines = f.readlines()
        poses = [] # frame_idx, pose x y z qx qy qz qw
        for line in lines:
            # Each line has 17 numbers, the first number is an integer denoting the frame index. The rest is a 4x4 matrix denoting the rigid body transform from the rectified perspective camera coordinates to the world coordinate system.
            pose = line.split()
            frame_idx = int(pose[0])
            cam0_to_world = np.array(pose[1:], dtype=np.float32).reshape(4, 4)
            translation = cam0_to_world[:3, 3]
            quaternion = tfm.quaternion_from_matrix(cam0_to_world)

            poses.append([frame_idx, translation, quaternion])
            
    return poses


if __name__ == '__main__':

    rospy.init_node('kitti360_data_reader', anonymous=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb_dir', type=str, default='/media/cc/Elements/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect')
    parser.add_argument('--depth_dir', type=str, default='/media/cc/Elements/KITTI-360/depth/2013_05_28_drive_0000_sync/sequences/0')
    parser.add_argument('--pose_txt', type=str, default='/media/cc/Elements/KITTI-360/data_poses/2013_05_28_drive_0000_sync/cam0_to_world.txt')
    parser.add_argument('--semantic_seg_dir', type=str, default='/media/cc/Elements/KITTI-360/data_2d_semantics/train/2013_05_28_drive_0000_sync/image_00/semantic_rgb')

    parser.add_argument('--rgb_image_topic', type=str, default='/kitti360/cam0/rgb')
    parser.add_argument('--depth_image_topic', type=str, default='/kitti360/cam0/depth')
    parser.add_argument('--camera_pose_topic', type=str, default='/kitti360/pose_cam')
    parser.add_argument('--semantic_seg_image_topic', type=str, default='/kitti360/cam0/semantic')

    parser.add_argument('--starting_frame_idx', type=int, default=1100)

    parser.add_argument('--loop_rate', type=int, default=0.5)
    parser.add_argument('--publish_semantic_seg', type=bool, default=True)

    args = parser.parse_args()

    rgb_image_pub = rospy.Publisher(args.rgb_image_topic, Image, queue_size=1)
    depth_image_pub = rospy.Publisher(args.depth_image_topic, Image, queue_size=1)
    camera_pose_pub = rospy.Publisher(args.camera_pose_topic, PoseStamped, queue_size=1)
    semantic_seg_image_pub = rospy.Publisher(args.semantic_seg_image_topic, Image, queue_size=1)

    # Read the pose data. Only publish the image with the corresponding pose
    pose_data = read_pose_txt(args.pose_txt)

    print("Number of poses = ", len(pose_data))

    # Loop through the pose data and publish the image and pose
    published_pose_idx = 0
    rate = rospy.Rate(args.loop_rate)
    for pose in pose_data:
        published_pose_idx += 1

        if published_pose_idx < args.starting_frame_idx:
            continue

        frame_idx, translation, quaternion = pose

        rgb_image_path = os.path.join(args.rgb_dir, str(frame_idx).zfill(10) + '.png')
        depth_image_path = os.path.join(args.depth_dir, str(frame_idx).zfill(10) + '.npy')

        rgb_image = cv2.imread(rgb_image_path)
        depth_image = np.load(depth_image_path)

        time = rospy.get_rostime()

        # Publish the image
        rgb_image_msg = Image()
        depth_image_msg = Image()
        if rgb_image is not None:
            rgb_image_msg = CvBridge().cv2_to_imgmsg(rgb_image)
            rgb_image_msg.header.stamp = time
            rgb_image_pub.publish(rgb_image_msg)
        else:
            # Raise an exception
            raise ValueError("RGB Image is None")
        
        if depth_image is not None:
            depth_image_msg = CvBridge().cv2_to_imgmsg(depth_image, encoding="32FC1")
            depth_image_msg.header.stamp = time
            depth_image_pub.publish(depth_image_msg)
        else:
            # Raise an exception
            raise ValueError("Depth Image is None")
        
        # Publish the camera pose
        if not init_pose_recorded:
            init_pose_recorded = True
            init_translation = translation
            init_quaternion = quaternion

        pose_msg = PoseStamped()
        pose_msg.header.stamp = time
        pose_msg.pose.position.x = translation[0] - init_translation[0]
        pose_msg.pose.position.y = translation[1] - init_translation[1]
        pose_msg.pose.position.z = translation[2] - init_translation[2]
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        camera_pose_pub.publish(pose_msg)

        if args.publish_semantic_seg:
            semantic_seg_image_path = os.path.join(args.semantic_seg_dir, str(frame_idx).zfill(10) + '.png')
            semantic_seg_image = cv2.imread(semantic_seg_image_path, cv2.IMREAD_UNCHANGED)

            semantic_seg_image_msg = Image()
            if semantic_seg_image is not None:
                semantic_seg_image_msg = CvBridge().cv2_to_imgmsg(semantic_seg_image)

                semantic_seg_image_msg.header.stamp = time
                semantic_seg_image_pub.publish(semantic_seg_image_msg)
            else:
                # Raise an exception
                raise ValueError("Semantic Segmentation Image is None")

        if published_pose_idx % 10 == 0:
            print("Progress: ", published_pose_idx, "/", len(pose_data))

        if rospy.is_shutdown():
            break

        rate.sleep()


    ## Test CODE. Load the depth image and visualize the point cloud
    # depth_image = np.load('/media/cc/Elements/KITTI-360/depth/2013_05_28_drive_0000_sync/sequences/0/0000001000.npy')

    # filtered_depth_image = cv2.bilateralFilter(depth_image, 5, 50, 50)

    # # plt.imshow(depth_image, cmap='gray')
    # # plt.colorbar()
    # # plt.show()


    # #788.629315 0.000000 687.158398 0.000000 786.382230 317.752196 0.000000 0.000000 0.000000

    # K=np.array([[788.629315, 0.000000, 687.158398],
    #             [0.000000, 786.382230, 317.752196],
    #             [0.000000, 0.000000, 1.000000]])
    
    # points = generate_point_cloud(filtered_depth_image, K)
    # visualize_point_cloud(points, 0.5)

