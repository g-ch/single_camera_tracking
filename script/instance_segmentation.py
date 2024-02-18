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
from mask_kpts_msgs.msg import Keypoint, MaskGroup, MaskKpts

class InstanceSegmentation:
    def __init__(self):
        # Mask2former
        # config = '../model/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
        # config = '/home/cc/git/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
        # script_path = os.path.realpath(__file__)
        # current_directory = os.path.dirname(script_path)
        # checkpoint = current_directory + '/../model/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

        config = '/home/cc/git/mmdetection/configs/mask2former/mask2former_r50_lsj_8x2_50e_coco.py'
        checkpoint = '/home/cc/chg_ws/ros_ws/semantic_map_ws/src/single_camera_tracking/model/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth'

        # Set the device to be used for evaluation
        device='cuda:0'
        self.model = self.load_model(config, checkpoint, device)

        # Set the confidence threshold
        self.confidence_threshold = 0.8
        self.concerned_labels = [2, 5, 7] # 2: car 5: bus 7: truck

        # Set the image subscriber
        self.image_sub = rospy.Subscriber("/coda/cam3/rgb", Image, self.image_callback)
        self.mask_pub = rospy.Publisher("/mask_group", MaskGroup, queue_size=1)

        # Set the bridge
        self.bridge = CvBridge()

        self.frame_id = "map"

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

    def result2mask(self, cv_image, result, show_result=True):
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

        # Remove the bboxes in high_confidence_idx_array that are not in the concerned_labels
        concerned_labels_idx_array = []
        for idx in high_confidence_idx_array:
            if labels[idx] in self.concerned_labels:
                concerned_labels_idx_array.append(idx)

        # Change the True/False values in the masks to 1/0
        masks = masks.astype(np.uint8)

        # Show the results by overlaying the masks as one RGB image
        if show_result:
            overlayed_mask = np.zeros([masks.shape[1], masks.shape[2], 3])
            for idx in concerned_labels_idx_array:
                # Set a random color for each mask
                color = np.random.randint(0, 255, size=(3,))
                # print("color = ", color)
                overlayed_mask[masks[idx, :, :] == 1] = color

            # draw bounding boxes
            for idx in concerned_labels_idx_array:
                bbox = bboxes[idx, :4].astype(np.int32)
                label_text = f'{labels[idx]}'
                # print("label_text = ", label_text)
                cv2.rectangle(overlayed_mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                cv2.putText(overlayed_mask, label_text, (bbox[0], bbox[1] - 2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            # Show the RGB image
            overlayed_mask = overlayed_mask.astype(np.uint8)
            overlayed_image = cv2.addWeighted(cv_image, 0.5, overlayed_mask, 0.5, 0)
            cv2.imshow("overlayed_image", overlayed_image)
            cv2.waitKey(1)

        # Publish the results
        mask_group = MaskGroup()
        # print("high confidence idx num = ", len(high_confidence_idx_array))
        for idx in concerned_labels_idx_array:
            this_mask = MaskKpts()
            this_mask.bbox_tl.x = bboxes[idx, 0]
            this_mask.bbox_tl.y = bboxes[idx, 1]
            this_mask.bbox_br.x = bboxes[idx, 2]
            this_mask.bbox_br.y = bboxes[idx, 3]
            this_mask.label = str(labels[idx])
            this_mask.mask = self.bridge.cv2_to_imgmsg(masks[idx, :, :], encoding="mono8")
            mask_group.header.stamp = rospy.Time.now()
            mask_group.header.frame_id = self.frame_id
            mask_group.objects.append(this_mask)
        
        self.mask_pub.publish(mask_group)

    def image_callback(self, msg):
        # Convert the image to OpenCV format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg) #desired_encoding="bgr8"
        except CvBridgeError as e:
            print(e)

        self.frame_id = msg.header.frame_id

        # Run inference and calculate the time
        time_start = time.time()
        result = self.inference(cv_image, show=False)
        self.result2mask(cv_image, result)
        time_end = time.time()
        print("Inference time (ms) = ", (time_end - time_start) * 1000)
        
    
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

    


    

