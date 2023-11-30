/**
 * @file tracking.cpp
 * @author Clarence Chen (g-ch@github.com)
 * @brief Using SuperPoint and SuperGlue to track objects and output the keypoints of the objects
 * @version 0.1
 * @date 2023-10-2
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include <memory>
#include <chrono>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <mask_kpts_msgs/MaskGroup.h>
#include <mask_kpts_msgs/MaskKpts.h>
#include <mask_kpts_msgs/Keypoint.h>
#include <unordered_set>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseStamped.h>
#include <yaml-cpp/yaml.h>


// define camera intrinsic parameters.
float c_camera_fx = 725.0087f; ///< Focal length in x direction. Unit: pixel
float c_camera_fy = 725.0087f; ///< Focal length in y direction. Unit: pixel
float c_camera_cx = 620.5f; ///< Principal point in x direction. Unit: pixel
float c_camera_cy = 187.f; ///< Principal point in y direction. Unit: pixel

float points_too_far_threshold = 30.f; ///< Threshold to remove points that are too far away from the camera. Unit: meter
float points_too_close_threshold = 0.5f; ///< Threshold to remove points that are too close to the camera. Unit: meter

int c_vote_number_threshold = 3;
float c_iou_threshold = 0.5f;
double c_bbox_size_threshold = 1000.0; // minimum size of bounding box to track


class BoundingBox{
public:
    BoundingBox()
    : x_min_(std::numeric_limits<double>::max()), 
      x_max_(std::numeric_limits<double>::min()), 
      y_min_(std::numeric_limits<double>::max()), 
      y_max_(std::numeric_limits<double>::min()){
    }
    ~BoundingBox(){}

    void setByTLAndBR(double tl_x, double tl_y, double br_x, double br_y){
        x_min_ = tl_x;
        y_min_ = tl_y;
        x_max_ = br_x;
        y_max_ = br_y;
    }

    double calculateIOU(const BoundingBox &other){
        double x_min = std::max(x_min_, other.x_min_);
        double y_min = std::max(y_min_, other.y_min_);
        double x_max = std::min(x_max_, other.x_max_);
        double y_max = std::min(y_max_, other.y_max_);

        double intersection_area = std::max(0.0, x_max - x_min) * std::max(0.0, y_max - y_min);
        double union_area = (x_max_ - x_min_) * (y_max_ - y_min_) + (other.x_max_ - other.x_min_) * (other.y_max_ - other.y_min_) - intersection_area;
        if(union_area == 0.0){
            return 0;
        }
        return intersection_area / union_area;
    }

    double size(){
        return (x_max_ - x_min_) * (y_max_ - y_min_);
    }

    double x_min_, x_max_, y_min_, y_max_;
};

class TrackingNode{
public:
    TrackingNode(){
        initialization();
    };
    ~TrackingNode(){};

private:
    ros::NodeHandle nh_;
    ros::Subscriber raw_image_sub_, segmentation_result_sub_, camera_pose_sub_;
    ros::Publisher image_pub_, mask_pub_, key_points_pub_;

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperGlue> superglue_;

    cv::Mat img_last_;
    cv::Mat depth_img_last_;
    cv::Mat depth_img_curr_;

    bool matched_points_ready_;

    int width_, height_; // image size for super glue
    
    std::vector<cv::KeyPoint> keypoints_last_ori_img_; // keypoints of last frame in real-size image coordinate
    std::vector<cv::KeyPoint> keypoints_curr_ori_img_; // keypoints of this frame in real-size image coordinate
    
    std::vector<int> tracking_ids_last_frame_; // track id of keypoints in the last frame
    std::vector<int> tracking_ids_curr_frame_; // track id of keypoints in this frame

    std::vector<int> track_ids_masks_last_; // track id of masks in the last frame

    std::vector<BoundingBox> bounding_boxes_last_frame_; // bounding boxes of objects in the last frame
    std::vector<BoundingBox> bounding_boxes_curr_frame_; // bounding boxes of objects in this frame
    
    int next_tracking_id_; // next track id


    std::vector<cv::DMatch> superglue_matches_; // superglue matches

    // define a color map to show different track ids
    std::vector<cv::Scalar> color_map_;

    Eigen::Vector3d camera_position_curr_, camera_position_last_;
    Eigen::Quaterniond camera_orientation_curr_, camera_orientation_last_;


    /// @brief Image callback function
    /// @param rgb_msg 
    /// @param depth_msg
    void imageCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg, const geometry_msgs::PoseStampedConstPtr& camera_pose_msg){
        cv_bridge::CvImagePtr cv_ptr, cv_ptr_depth;
        try{
            cv_ptr = cv_bridge::toCvCopy(rgb_msg, sensor_msgs::image_encodings::RGB8);
            cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Get camera pose
        camera_position_curr_ = Eigen::Vector3d(camera_pose_msg->pose.position.x, camera_pose_msg->pose.position.y, camera_pose_msg->pose.position.z);
        camera_orientation_curr_ = Eigen::Quaterniond(camera_pose_msg->pose.orientation.w, camera_pose_msg->pose.orientation.x, camera_pose_msg->pose.orientation.y, camera_pose_msg->pose.orientation.z);

        // Add a timer
        auto start = std::chrono::high_resolution_clock::now();

        // Get image
        cv::Mat img = cv_ptr->image;
        depth_img_curr_ = cv_ptr_depth->image;

        // Convert RGB to grayscale and do inference
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        superglueInference(img);

        // End of timer
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference cost " << duration / 1000 << " MS" << std::endl;
    }


    /// @brief The function is used to get superpoint and superglue inference result
    /// @param img 
    void superglueInference(cv::Mat &img, bool draw_match = true){
        if(matched_points_ready_){
            std::cout << "Last Matched result not used. Will skip this frame." << std::endl;
            return;
        }

        // Resize image
        int img_ori_width = img.cols;
        int img_ori_height = img.rows;
        cv::resize(img, img, cv::Size(width_, height_));

        // infer superpoint
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points_curr;
        static Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points_last;
        superpoint_->infer(img, feature_points_curr);

        // Find keypoint correspondences
        static std::vector<cv::KeyPoint> keypoints_last;
        std::vector<cv::KeyPoint> keypoints_curr;
        keypoints_curr_ori_img_.clear();
        for(size_t i = 0; i < feature_points_curr.cols(); ++i){
            double score = feature_points_curr(0, i);
            double x = feature_points_curr(1, i);
            double y = feature_points_curr(2, i);
            keypoints_curr.emplace_back(x, y, 8, -1, score);

            // Convert to real-size image coordinate
            x = x / width_ * img_ori_width;
            y = y / height_ * img_ori_height;
            keypoints_curr_ori_img_.emplace_back(x, y, 8, -1, score);
        }

        static cv::Mat depth_img_last_exchange;
        static Eigen::Vector3d camera_position_last_exchange;
        static Eigen::Quaterniond camera_orientation_last_exchange;

        // Check if it is the first frame
        static bool first_frame = true;
        if(first_frame){
            first_frame = false;
            img_last_ = img;
            feature_points_last = feature_points_curr;
            keypoints_last = keypoints_curr;
            depth_img_last_exchange = depth_img_curr_; // Set the first depth image to the last depth image
            camera_position_last_exchange = camera_position_curr_;
            camera_orientation_last_exchange = camera_orientation_curr_;
            return;
        }

        // infer superglue
        std::vector<cv::DMatch> superglue_matches;
        superglue_->matching_points(feature_points_last, feature_points_curr, superglue_matches);
        superglue_matches_ = superglue_matches;

        // Set keypoints_last_ori_img_
        keypoints_last_ori_img_.clear();
        for(size_t i = 0; i < feature_points_last.cols(); ++i){
            double score = feature_points_last(0, i);
            double x = feature_points_last(1, i);
            double y = feature_points_last(2, i);
            // Convert to real-size image coordinate
            x = x / width_ * img_ori_width;
            y = y / height_ * img_ori_height;
            keypoints_last_ori_img_.emplace_back(x, y, 8, -1, score);
        }

        // draw matches
        if(draw_match)
        {
            cv::Mat match_image;
            cv::drawMatches(img_last_, keypoints_last, img, keypoints_curr, superglue_matches, match_image);

            // Publish image
            cv_bridge::CvImage cv_image;
            cv_image.image = match_image;
            cv_image.encoding = "bgr8";
            image_pub_.publish(cv_image.toImageMsg());

            // Save image and feature points
            cv::imshow("match_image", match_image);
            cv::waitKey(10);
        }

        depth_img_last_ = depth_img_last_exchange; // Set the last depth image to the depth image of the last frame
        depth_img_last_exchange = depth_img_curr_;

        camera_position_last_ = camera_position_last_exchange;
        camera_position_last_exchange = camera_position_curr_;
        camera_orientation_last_ = camera_orientation_last_exchange;
        camera_orientation_last_exchange = camera_orientation_curr_;

        img_last_ = img;
        feature_points_last = feature_points_curr;
        keypoints_last = keypoints_curr;
        matched_points_ready_ = true;
    }

    /// @brief The function is used to callback segmentation result
    void segmentationResultCallback(const mask_kpts_msgs::MaskGroup& msg){
        if(!matched_points_ready_){
            std::cout << "No matched points ready !!!!!!!!!" << std::endl;
            return;
        }

        if(msg.objects.size() == 0){
            std::cout << "No mask received !!!!!!!!!" << std::endl;
            matched_points_ready_ = false;
            // Publish the original message
            mask_kpts_msgs::MaskGroup copied_msg = msg;
            copied_msg.header.stamp = ros::Time::now();
            mask_pub_.publish(copied_msg);
            return;
        }

        // Copy the message to a local variable
        mask_kpts_msgs::MaskGroup copied_msg = msg;

        // Create a vector to store all the masks
        std::vector<cv::Mat> masks;
        std::cout << "msg.objects.size() = " << msg.objects.size() << std::endl;
        for(size_t i = 0; i < msg.objects.size(); ++i){
            cv_bridge::CvImagePtr cv_ptr;
            try{
                cv_ptr = cv_bridge::toCvCopy(msg.objects[i].mask, sensor_msgs::image_encodings::MONO8);
            }
            catch (cv_bridge::Exception& e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            masks.push_back(cv_ptr->image);
        }

        // Set bounding boxes
        bounding_boxes_curr_frame_.clear();
        for(size_t i = 0; i < msg.objects.size(); ++i){
            BoundingBox bounding_box;
            bounding_box.setByTLAndBR(msg.objects[i].bbox_tl.x, msg.objects[i].bbox_tl.y, msg.objects[i].bbox_br.x, msg.objects[i].bbox_br.y);
            bounding_boxes_curr_frame_.emplace_back(bounding_box);
        }

        // Iterate all masks. Find the keypoints in each mask and store their indices.
        std::vector<std::vector<int>> keypoints_in_masks(masks.size());
        for (int i = 0; i < keypoints_curr_ori_img_.size(); i++) {
            const auto& kp = keypoints_curr_ori_img_[i];
            int pt_y = round(kp.pt.y);
            int pt_x = round(kp.pt.x);
            if(pt_y < 0 || pt_y >= masks[0].rows || pt_x < 0 || pt_x >= masks[0].cols){
                continue;
            }
            for (int m = 0; m < masks.size(); m++) {
                if (masks[m].at<uchar>(pt_y, pt_x) > 0) { // Each mask is a cv::Mat of type CV_8UC1, with non-zero pixels indicating the object.
                    keypoints_in_masks[m].push_back(i);
                    break; // A keypoint cannot be in multiple masks.
                }
            }
        }

        std::cout << "keypoints_in_masks.size() = " << keypoints_in_masks.size() << std::endl;

        // Initialize tracking_ids_curr_frame_ with -1
        tracking_ids_curr_frame_ = std::vector<int>(keypoints_curr_ori_img_.size(), -1);
        std::vector<int> track_ids_masks(masks.size(), -1);
            
        if(tracking_ids_last_frame_.empty()){ //first frame pair
            std::cout << "First frame pair. keypoints_last_ori_img_.size() = " << keypoints_last_ori_img_.size() << std::endl;
            tracking_ids_last_frame_ = std::vector<int>(keypoints_last_ori_img_.size(), -1);
        }

        if(tracking_ids_last_frame_.size() != keypoints_last_ori_img_.size()){
            std::cout << "tracking_ids_last_frame_.size() = " << tracking_ids_last_frame_.size() << std::endl;
            return;
        }

        // Set a matrix to quickly find the matched keypoints
        cv::Mat match_matrix = cv::Mat::zeros(keypoints_curr_ori_img_.size(), keypoints_last_ori_img_.size(), CV_8U);
        for(const cv::DMatch& match : superglue_matches_) {
            match_matrix.at<uchar>(match.trainIdx, match.queryIdx) = 1;
        }

        pcl::PointCloud<pcl::PointXYZRGB> key_points;
        Eigen::Matrix3d camera_orientation_matrix_curr_ = camera_orientation_curr_.toRotationMatrix();
        Eigen::Matrix3d camera_orientation_matrix_last_ = camera_orientation_last_.toRotationMatrix();

        // Iterate all masks. Find the keypoints in each mask and store their indices.
        std::unordered_set<int> matched_track_ids;
        for (int m = 0; m < masks.size(); m++) {
            std::cout << "mask " << m << " total mask size = " << masks.size() << std::endl;
            
            //Ignore the mask if it is too small.
            if(bounding_boxes_curr_frame_[m].size() < c_bbox_size_threshold){
                std::cout << "mask " << m << " is too small. Ignore it." << std::endl;
                track_ids_masks[m] = -1;
                continue;
            }

            int mask_b_color = (m * 50 + 50) % 255; // Set a color b channel for each mask

            // Vote for the tracking ID with the most matched keypoints.
            std::map<int, int> id_votes;
            for (int i : keypoints_in_masks[m]) {
                for (int j = 0; j < keypoints_last_ori_img_.size(); j++) {
                    if (match_matrix.at<uchar>(i, j) > 0) { // If the curr keypoint is matched to a last keypoint.
                        int tracking_id = tracking_ids_last_frame_[j];
                        if (tracking_id >= 0) {
                            id_votes[tracking_id]++;
                        }

                        // Storing matched keypoints for each mask. Use depth um image and camera intrinsic parameters to calculate the 3D position of the keypoints.
                        /// TODO: Remove the points with invalid depth.
                        Eigen::Vector3d p_curr_global, p_last_global;
                        p_curr_global[0] = keypoints_curr_ori_img_[i].pt.x;
                        p_curr_global[1] = keypoints_curr_ori_img_[i].pt.y;
                        p_curr_global[2] = depth_img_curr_.at<float>(p_curr_global[1], p_curr_global[0]);
                        p_curr_global[0] = (p_curr_global[0] - c_camera_cx) * p_curr_global[2] / c_camera_fx;
                        p_curr_global[1] = (p_curr_global[1] - c_camera_cy) * p_curr_global[2] / c_camera_fy;
                        p_curr_global = camera_orientation_matrix_curr_ * p_curr_global + camera_position_curr_;

                        p_last_global[0] = keypoints_last_ori_img_[j].pt.x;
                        p_last_global[1] = keypoints_last_ori_img_[j].pt.y;
                        p_last_global[2] = depth_img_last_.at<float>(p_last_global[1], p_last_global[0]);
                        p_last_global[0] = (p_last_global[0] - c_camera_cx) * p_last_global[2] / c_camera_fx;
                        p_last_global[1] = (p_last_global[1] - c_camera_cy) * p_last_global[2] / c_camera_fy;
                        p_last_global = camera_orientation_matrix_last_ * p_last_global + camera_position_last_;

                        // Check if the point is too far away from the camera. If so, don't use it for t_matrix estimation.
                        if((p_curr_global - camera_position_curr_).norm() > points_too_far_threshold || (p_last_global - camera_position_last_).norm() > points_too_far_threshold){
                            // Even if too far, it's still useful to track the object. So don't decrease the vote.
                            // id_votes[tracking_id]--; // If the point is too far away, decrease the vote.
                            continue;
                        }

                        // Add the matched keypoints to the message to be published for map building
                        mask_kpts_msgs::Keypoint kpt_curr, kpt_last;
                        kpt_curr.x = p_curr_global[0];
                        kpt_curr.y = p_curr_global[1];
                        kpt_curr.z = p_curr_global[2];
                        kpt_last.x = p_last_global[0];
                        kpt_last.y = p_last_global[1];
                        kpt_last.z = p_last_global[2];
                        copied_msg.objects[m].kpts_curr.push_back(kpt_curr);
                        copied_msg.objects[m].kpts_last.push_back(kpt_last);

                        // Visualize the matched keypoints
                        pcl::PointXYZRGB p_curr, p_last;
                        p_curr.x = kpt_curr.x;
                        p_curr.y = kpt_curr.y;
                        p_curr.z = kpt_curr.z;
                        p_curr.r = 255; 
                        p_curr.g = 0;
                        p_curr.b = mask_b_color; // Set a color for each mask

                        p_last.x = kpt_last.x;
                        p_last.y = kpt_last.y;
                        p_last.z = kpt_last.z;
                        p_last.r = 0;
                        p_last.g = 255; 
                        p_last.b = mask_b_color; // Set a color for each mask

                        key_points.push_back(p_curr);
                        key_points.push_back(p_last);

                        // Show kpt_curr and kpt_last
                        // std::cout << "kpt_curr = " << kpt_curr.x << ", " << kpt_curr.y << ", " << kpt_curr.z << std::endl;
                        // std::cout << "kpt_last = " << kpt_last.x << ", " << kpt_last.y << ", " << kpt_last.z << std::endl;

                        break;  // Breaking the loop as one point can only have one match.
                    }
                }
            }

            // Find the tracking ID with the most votes.
            int best_tracking_id = -1;
            int best_votes = 0;
            for (const auto& [id, votes] : id_votes) {
                if (votes > best_votes) {
                    best_tracking_id = id;
                    best_votes = votes;
                }
            }

            std::cout << "best_tracking_id = " << best_tracking_id << ", best_votes = " << best_votes << std::endl;

            // Decide whether to use the best existing tracking ID or a new one.
            if (best_tracking_id >=0 && best_votes >= c_vote_number_threshold && matched_track_ids.find(best_tracking_id) == matched_track_ids.end()) {
                track_ids_masks[m] = best_tracking_id;
                matched_track_ids.insert(best_tracking_id);
                std::cout << "tracked object = " << best_tracking_id << std::endl;
            } 
            else{
                // Try to match by IOU with the bounding boxes of the last frame in case too less features are matched.
                for(int i = 0; i < bounding_boxes_last_frame_.size(); ++i){
                    double iou = bounding_boxes_curr_frame_[m].calculateIOU(bounding_boxes_last_frame_[i]);
                    if(iou > c_iou_threshold){
                        int id = track_ids_masks_last_[i];
                        if(matched_track_ids.find(id) == matched_track_ids.end()){
                            std::cout << "IOU tracked object = " << id << std::endl;
                            if(id < 0){
                                id = next_tracking_id_;
                                next_tracking_id_ ++;
                            }
                            track_ids_masks[m] = id;
                            matched_track_ids.insert(id);
                            std::cout << "IOU Make it a new tracking id = " << id << std::endl;
                            break;
                        }
                    }
                }
            }

            // Set the track id of the mask to the keypoints in the mask.
            for (int i : keypoints_in_masks[m]) {
                tracking_ids_curr_frame_[i] = track_ids_masks[m];
            }
        }

        // Publish the key points for visualization
        sensor_msgs::PointCloud2 key_points_msg;
        pcl::toROSMsg(key_points, key_points_msg);
        key_points_msg.header.stamp = ros::Time::now();
        key_points_msg.header.frame_id = "map";
        key_points_pub_.publish(key_points_msg);

        // Publish the copied message
        copied_msg.header.stamp = ros::Time::now();
        for(size_t i = 0; i < masks.size(); ++i){
            copied_msg.objects[i].track_id = track_ids_masks[i];
        }
        mask_pub_.publish(copied_msg);

        // Show the result in a single image
        showResult(masks, bounding_boxes_curr_frame_, track_ids_masks, copied_msg);

        // Update for the next iteration.
        tracking_ids_last_frame_ = tracking_ids_curr_frame_;
        matched_points_ready_ = false;
        bounding_boxes_last_frame_ = bounding_boxes_curr_frame_;
        track_ids_masks_last_ = track_ids_masks;

        std::cout << "segmentationResultCallback finished" << std::endl;
        return;
    }

    /// @brief The function is used to show the tracking result in a single image
    void showResult(const std::vector<cv::Mat> &masks, const std::vector<BoundingBox> &bboxes, const std::vector<int> &track_ids_masks, const mask_kpts_msgs::MaskGroup &msg)
    {
        // Show the masks and matched curr keypoints in one image
        cv::Mat mask_image = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC3);
        for(size_t i = 0; i < masks.size(); ++i){
            cv::Mat mask = masks[i];
            cv::Mat mask_color = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);

            // Get a color from the color map for each track id
            int track_id = track_ids_masks[i];
            cv::Scalar color;
            if(track_id < 0){
                color = cv::Scalar(255, 255, 255);  
            }else{
                color = color_map_[track_id % 256];
            }

            mask_color.setTo(color, mask);
            mask_image = mask_image + mask_color;

            // Add bounding box
            cv::rectangle(mask_image, cv::Point(bboxes[i].x_min_, bboxes[i].y_min_), cv::Point(bboxes[i].x_max_, bboxes[i].y_max_), color, 1);

            // Add matched curr keypoints
            cv::Scalar reversed_color = cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]);
            for(const auto &kpt : msg.objects[i].kpts_curr){
                cv::circle(mask_image, cv::Point(kpt.x, kpt.y), 2, reversed_color, -1);
            }

            // Find the center of the matched curr keypoints and write the track id
            int x_sum = 0, y_sum = 0;
            for(const auto &kpt : msg.objects[i].kpts_curr){
                x_sum += kpt.x;
                y_sum += kpt.y;
            }
            if(x_sum != 0 && y_sum != 0){
                int x_center = x_sum / msg.objects[i].kpts_curr.size();
                int y_center = y_sum / msg.objects[i].kpts_curr.size();
                cv::putText(mask_image, std::to_string(track_id), cv::Point(x_center, y_center), cv::FONT_HERSHEY_SIMPLEX, 0.8, reversed_color, 2);
            }
        }

        cv::imshow("mask_image", mask_image);
        cv::waitKey(10);
    }


    /// @brief The function is used for initializating the node
    void initialization(){
        //raw_image_sub_ = nh_.subscribe("/camera_rgb_image", 1, &TrackingNode::imageCallback, this);
        
        // Subscribe to camera_rgb_image, camera_depth_image and camera pose synchronously
        message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub(nh_, "/camera_rgb_image", 1);
        message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, "/camera_depth_image", 1);
        message_filters::Subscriber<geometry_msgs::PoseStamped> camera_pose_sub(nh_, "/camera_pose", 1);
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_image_sub, depth_image_sub, camera_pose_sub);
        sync.registerCallback(boost::bind(&TrackingNode::imageCallback, this, _1, _2, _3));

        segmentation_result_sub_ = nh_.subscribe("/mask_group", 1, &TrackingNode::segmentationResultCallback, this);

        image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/image_super_glued", 1);
        mask_pub_ = nh_.advertise<mask_kpts_msgs::MaskGroup>("/mask_group_super_glued", 1);
        key_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/key_points", 1);

        std::string package_path = ros::package::getPath("single_camera_tracking");
        std::string config_path = package_path + "/SuperPoint-SuperGlue-TensorRT/config/config.yaml";
        std::string model_dir = package_path + "/SuperPoint-SuperGlue-TensorRT/weights/";
        Configs configs(config_path, model_dir);

        height_ = configs.superglue_config.image_height;
        width_ = configs.superglue_config.image_width; 

        next_tracking_id_ = 1; //Start from 1. CHG
        matched_points_ready_ = false;

        // Set a random color map for visualization
        for(int i=0; i<256; ++i){
            cv::RNG rng(i);
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            color_map_.push_back(color);
        }

        // Create superpoint detector and superglue matcher. Build engine
        std::cout << "Building inference engine......" << std::endl;
        superpoint_ = std::make_shared<SuperPoint>(configs.superpoint_config);
        if (!superpoint_->build()){
            std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
            return;
        }
        superglue_ = std::make_shared<SuperGlue>(configs.superglue_config);
        if (!superglue_->build()){
            std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
            return;
        }
        std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

        // test();

        ros::spin();
    }

    void test(){
        // load image
        cv::Mat image0 = cv::imread("/home/clarence/git/SuperPoint-SuperGlue-TensorRT/data/1/rgb_00365.jpg", cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread("/home/clarence/git/SuperPoint-SuperGlue-TensorRT/data/1/rgb_00366.jpg", cv::IMREAD_GRAYSCALE);

        superglueInference(image0);
        superglueInference(image1);
    }
};

int main(int argc, char** argv){
    ros::init(argc, argv, "tracking_node");
    
    // Read parameters from yaml file
    std::string package_path = ros::package::getPath("single_camera_tracking");
    std::string config_path = package_path + "/cfg/settings.yaml";

    YAML::Node config = YAML::LoadFile(config_path);
    c_camera_fx = config["camera_fx"].as<float>();
    c_camera_fy = config["camera_fy"].as<float>();
    c_camera_cx = config["camera_cx"].as<float>();
    c_camera_cy = config["camera_cy"].as<float>();
    points_too_far_threshold = config["points_too_far_threshold"].as<float>();
    points_too_close_threshold = config["points_too_close_threshold"].as<float>();
    c_vote_number_threshold = config["vote_number_threshold"].as<int>();
    c_iou_threshold = config["iou_threshold"].as<float>();
    c_bbox_size_threshold = config["bbox_size_threshold"].as<double>();

    // Create a node
    TrackingNode tracking_node;
    
    return 0;
}
