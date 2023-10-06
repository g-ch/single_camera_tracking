#!/usr/bin/env python
#coding:utf-8

import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector

import os
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import time
import cv2
import numpy as np
import torch
from single_camera_tracking.msg import Keypoint, MaskGroup, MaskKpts

class InstanceSegmentation:
    def __init__(self):
        # Mask2former
        config = '/home/clarence/git/openmmlab/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
        checkpoint = '/home/clarence/git/openmmlab/mmdetection/mymodels/mask2former/ins_resnet_50/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

        # Set the device to be used for evaluation
        device='cuda:0'
        self.model = self.load_model(config, checkpoint, device)

        # Set the confidence threshold
        self.confidence_threshold = 0.4
        self.result_folder = "/home/clarence/ros_ws/semantic_dsp_ws/src/single_camera_tracking/data/result"

        # Set the image subscriber
        self.image_sub = rospy.Subscriber("/camera_rgb_image", Image, self.image_callback)
        self.mask_pub = rospy.Publisher("/mask_group", MaskGroup, queue_size=1)

        # Set the bridge
        self.bridge = CvBridge()

        # Spin
        rospy.spin()

    #Function to load the model
    def load_model(self, config, checkpoint, device='cuda:0'):
        # Load the config
        config = mmcv.Config.fromfile(config)

        # Initialize the detector
        model = build_detector(config.model)

        # Load checkpoint
        checkpoint = load_checkpoint(model, checkpoint, map_location=device)

        # Set the classes of models for inference
        model.CLASSES = checkpoint['meta']['CLASSES']

        # We need to set the model's cfg for inference
        model.cfg = config

        # Convert the model to GPU
        model.to(device)
        # Convert the model into evaluation mode
        model.eval()

        return model
    
    def inference(self, img, show=False):
        # Run inference
        result = inference_detector(self.model, img)
        if show:
            self.visualize(img, result)
        
        return result
    
    def visualize(self, img, result):
        # Show the results
        show_result_pyplot(self.model, img, result, score_thr=0.3)

    def result2mask(self, result):
        # Process the results
        assert isinstance(result, tuple)
        bbox_result, mask_result = result
        bboxes = np.vstack(bbox_result)

        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)

        if len(labels) == 0:
            bboxes = np.zeros([0, 5])
            masks = np.zeros([0, 0, 0])
        # draw segmentation masks
        else:
            masks = mmcv.concat_list(mask_result)

            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=0).detach().cpu().numpy()
            else:
                masks = np.stack(masks, axis=0)
            # dummy bboxes
            if bboxes[:, :4].sum() == 0:
                num_masks = len(bboxes)
                x_any = masks.any(axis=1)
                y_any = masks.any(axis=2)
                for idx in range(num_masks):
                    x = np.where(x_any[idx, :])[0]
                    y = np.where(y_any[idx, :])[0]
                    if len(x) > 0 and len(y) > 0:
                        bboxes[idx, :4] = np.array(
                            [x[0], y[0], x[-1] + 1, y[-1] + 1],
                            dtype=np.float32)

        # Remove the bboxes with low confidence
        high_confidence_idx_array = np.where(bboxes[:, -1] > self.confidence_threshold)[0]
        print(high_confidence_idx_array)

        # Change the True/False values in the masks to 255/0
        masks = masks.astype(np.uint8)
        masks *= 255

        # Publish the results
        mask_group = MaskGroup()
        for idx in high_confidence_idx_array:
            this_mask = MaskKpts()
            this_mask.label = str(labels[idx])
            this_mask.mask = masks[idx, :, :].flatten().tolist()
            mask_group.header.stamp = rospy.Time.now()
            mask_group.objects.append(this_mask)
            self.mask_pub.publish(mask_group)

    def image_callback(self, msg):
        # Convert the image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # Run inference
        result = self.inference(cv_image, show=False)
        print("Got inference result!")
        self.result2mask(result)

        
    
# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('instance_segmentation', anonymous=True)

    # Initialize the class
    inst_seg = InstanceSegmentation()

    # # Load the image
    # img = '/home/clarence/git/SuperPoint-SuperGlue-TensorRT/data/1/rgb_00365.jpg'
    # img = cv2.imread(img)

    # # Run inference TEST
    # result = inst_seg.inference(img, show=True)
    # inst_seg.result2mask(result)

    


    

