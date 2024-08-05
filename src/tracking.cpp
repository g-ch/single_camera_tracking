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
#include <iomanip>

#include "object_info_handler.h"

// define camera intrinsic parameters with default values
float c_camera_fx = 569.8286f; ///< Focal length in x direction. Unit: pixel
float c_camera_fy = 565.4818f; ///< Focal length in y direction. Unit: pixel
float c_camera_cx = 439.2660f; ///< Principal point in x direction. Unit: pixel
float c_camera_cy = 360.5810f; ///< Principal point in y direction. Unit: pixel

Eigen::Matrix3d camera_intrinsic_matrix = Eigen::Matrix3d::Identity();
Eigen::Matrix3d camera_intrinsic_matrix_inv = Eigen::Matrix3d::Identity();


float points_too_far_threshold = 20.f; ///< Threshold to remove points that are too far away from the camera. Unit: meter
float points_too_close_threshold = 0.5f; ///< Threshold to remove points that are too close to the camera. Unit: meter

int c_vote_number_threshold = 3;
float c_iou_threshold = 0.5f;
double c_bbox_size_threshold = 1000.0; // minimum size of bounding box to track

std::string rgb_image_topic, depth_image_topic, camera_pose_topic;

ObjectInfoHandler object_info_handler;

/**
 * @brief A class to store the bounding box of an object and do IOU calculation
 * **/
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


/**
 * @brief Main class to track objects with ROS interface
 * **/
class TrackingNode{
public:
    TrackingNode(bool if_semantic_msg_available=false){
        if(if_semantic_msg_available){
            runWithSemanticImage();
        }else{
            run();
        }
    };
    ~TrackingNode(){};

private:
    ros::NodeHandle nh_;
    ros::Subscriber raw_image_sub_, segmentation_result_sub_, camera_pose_sub_;
    ros::Publisher image_pub_, mask_pub_, key_points_pub_;
    ros::Publisher depth_image_repub_, camera_pose_repub_;
    ros::Publisher original_point_cloud_pub_;

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

    uint32_t seq_id_; // frame id

    std::vector<cv::DMatch> superglue_matches_; // superglue matches

    // define a color map to show different track ids
    std::vector<cv::Scalar> color_map_;

    Eigen::Vector3d camera_position_curr_, camera_position_last_;
    Eigen::Quaterniond camera_orientation_curr_, camera_orientation_last_;


    /// @brief Image callback function
    /// @param rgb_msg 
    /// @param depth_msg
    void imageCallback(const sensor_msgs::ImageConstPtr& rgb_msg, const sensor_msgs::ImageConstPtr& depth_msg, const geometry_msgs::PoseStampedConstPtr& camera_pose_msg){
        std::cout << "Image callback" << std::endl;

        cv_bridge::CvImagePtr cv_ptr, cv_ptr_depth;
        try{
            cv_ptr = cv_bridge::toCvCopy(rgb_msg); //sensor_msgs::image_encodings::RGB8
            cv_ptr_depth = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);

            seq_id_ = rgb_msg->header.seq;
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

        // Publish the original PointCloud for visualization
        pcl::PointCloud<pcl::PointXYZRGB> original_point_cloud;
        for(int i = 0; i < cv_ptr_depth->image.rows; ++i){
            for(int j = 0; j < cv_ptr_depth->image.cols; ++j){
                pcl::PointXYZRGB point;
                Eigen::Vector3d pixel_position(j, i, 1);
                Eigen::Vector3d p = camera_intrinsic_matrix_inv * pixel_position * cv_ptr_depth->image.at<float>(i, j);
                p = camera_orientation_curr_.toRotationMatrix() * p + camera_position_curr_; // Transform the point to the global coordinate
                point.x = p[0];
                point.y = p[1];
                point.z = p[2];
                point.r = cv_ptr->image.at<cv::Vec3b>(i, j)[2];
                point.g = cv_ptr->image.at<cv::Vec3b>(i, j)[1];
                point.b = cv_ptr->image.at<cv::Vec3b>(i, j)[0];
                original_point_cloud.push_back(point);
            }
        }
        sensor_msgs::PointCloud2 original_point_cloud_msg;
        pcl::toROSMsg(original_point_cloud, original_point_cloud_msg);
        original_point_cloud_msg.header.stamp = ros::Time::now();
        original_point_cloud_msg.header.frame_id = "map";
        original_point_cloud_pub_.publish(original_point_cloud_msg);

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
    void instanceSegmentationResultCallback(const mask_kpts_msgs::MaskGroup& msg){
        sensor_msgs::ImageConstPtr semantic_seg_img;
        segmentationResultHandling(msg, semantic_seg_img, false);
    }

    /// @brief The function is used to callback segmentation result, with semantic segmentation image
    void panopticSegmentationResultCallback(const mask_kpts_msgs::MaskGroup::ConstPtr& msg, const sensor_msgs::ImageConstPtr& semantic_seg_img){

        std::cout << "Panoptic segmentation result callback" << std::endl;
        segmentationResultHandling(*msg, semantic_seg_img, true);
    }
    
    /// @brief Handling the segmentation result. Find the keypoints in each mask and publish the keypoints and masks
    /// @param msg MaskGroup message from instance segmentation
    /// @param semantic_seg_img Semantic segmentation image (optional)
    /// @param update_with_semantic_seg Flag to update the mask with semantic segmentation. If true, the mask will be updated with semantic segmentation.
    void segmentationResultHandling(const mask_kpts_msgs::MaskGroup& msg, const sensor_msgs::ImageConstPtr& semantic_seg_img, bool update_with_semantic_seg = false){
        if(msg.objects.size() == 0 || !matched_points_ready_){
            if(!matched_points_ready_){
                std::cout << "No matched points ready !!!!!!!!!" << std::endl;
            }else{
                std::cout << "No matched points ready !!!!!!!!!. msg.objects.size() = " << msg.objects.size() << std::endl;
            }

            matched_points_ready_ = false;

            // Publish the original message
            mask_kpts_msgs::MaskGroup copied_msg = msg;
            copied_msg.header.stamp = ros::Time::now();
            copied_msg.header.frame_id = "map";
            copied_msg.header.seq = seq_id_;
            mask_pub_.publish(copied_msg);

            // Publish depth_img_curr_
            cv_bridge::CvImage cv_depth_image(copied_msg.header, "32FC1", depth_img_curr_);
            depth_image_repub_.publish(cv_depth_image.toImageMsg());

            // Publish camera pose
            geometry_msgs::PoseStamped camera_pose_msg;
            camera_pose_msg.header = copied_msg.header;
            camera_pose_msg.pose.position.x = camera_position_curr_[0];
            camera_pose_msg.pose.position.y = camera_position_curr_[1];
            camera_pose_msg.pose.position.z = camera_position_curr_[2];
            camera_pose_msg.pose.orientation.w = camera_orientation_curr_.w();
            camera_pose_msg.pose.orientation.x = camera_orientation_curr_.x();
            camera_pose_msg.pose.orientation.y = camera_orientation_curr_.y();
            camera_pose_msg.pose.orientation.z = camera_orientation_curr_.z();
            camera_pose_repub_.publish(camera_pose_msg);

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

        // // Add the mask and bounding box to one image and show it. For debug.
        // cv::Mat seg_mask_image = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC1);
        // for(size_t i = 0; i < masks.size(); ++i){
        //     cv::Mat merged_img;
        //     cv::max(masks[i], seg_mask_image, merged_img);
        //     seg_mask_image = merged_img;
        // }
        // seg_mask_image = seg_mask_image * 255;
        // for(size_t i = 0; i < bounding_boxes_curr_frame_.size(); ++i){
        //     cv::rectangle(seg_mask_image, cv::Point(bounding_boxes_curr_frame_[i].x_min_, bounding_boxes_curr_frame_[i].y_min_), cv::Point(bounding_boxes_curr_frame_[i].x_max_, bounding_boxes_curr_frame_[i].y_max_), cv::Scalar(255), 1);
        // }
        // cv::imshow("seg_mask_image", seg_mask_image);

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

        // Create a point cloud to store the matched keypoints for visualization
        pcl::PointCloud<pcl::PointXYZRGB> key_points;

        // Get camera orientation matrix
        Eigen::Matrix3d camera_orientation_matrix_curr_ = camera_orientation_curr_.toRotationMatrix();
        Eigen::Matrix3d camera_orientation_matrix_last_ = camera_orientation_last_.toRotationMatrix();

        // Make a mask for invalid depth points
        cv::Mat invalid_depth_mask = cv::Mat::zeros(depth_img_curr_.rows, depth_img_curr_.cols, CV_8U);
        for(int i = 0; i < depth_img_curr_.rows; ++i){
            for(int j = 0; j < depth_img_curr_.cols; ++j){
                if(depth_img_curr_.at<float>(i, j) > points_too_far_threshold || depth_img_curr_.at<float>(i, j) < points_too_close_threshold){
                    invalid_depth_mask.at<uchar>(i, j) = 255;
                }
            }
        }

        // Make a mask for invalid depth points in the last frame
        cv::Mat invalid_depth_mask_last = cv::Mat::zeros(depth_img_last_.rows, depth_img_last_.cols, CV_8U);
        for(int i = 0; i < depth_img_last_.rows; ++i){
            for(int j = 0; j < depth_img_last_.cols; ++j){
                if(depth_img_last_.at<float>(i, j) > points_too_far_threshold || depth_img_last_.at<float>(i, j) < points_too_close_threshold){
                    invalid_depth_mask_last.at<uchar>(i, j) = 255;
                }
            }
        }

        cv::imshow("invalid_depth_mask tracking", invalid_depth_mask);

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

                        // Check if the point is too close or too far to the camera using the depth mask. If so, don't use it for t_matrix estimation.
                        if(invalid_depth_mask.at<uchar>(keypoints_curr_ori_img_[i].pt.y, keypoints_curr_ori_img_[i].pt.x) > 0){
                            continue;
                        }
                        if(invalid_depth_mask_last.at<uchar>(keypoints_last_ori_img_[j].pt.y, keypoints_last_ori_img_[j].pt.x) > 0){
                            continue;
                        }

                        // Storing matched keypoints for each mask. Use depth um image and camera intrinsic parameters to calculate the 3D position of the keypoints.
                        Eigen::Vector3d p_curr_global, p_last_global;

                        // Use camera_intrinsic_matrix to calculate the 3D position of the keypoints.
                        Eigen::Vector3d pixel_position_current(keypoints_curr_ori_img_[i].pt.x, keypoints_curr_ori_img_[i].pt.y, 1);
                        p_curr_global = camera_intrinsic_matrix_inv * pixel_position_current * depth_img_curr_.at<float>(keypoints_curr_ori_img_[i].pt.y, keypoints_curr_ori_img_[i].pt.x);
                        p_curr_global = camera_orientation_matrix_curr_ * p_curr_global + camera_position_curr_;


                        Eigen::Vector3d pixel_position_last(keypoints_last_ori_img_[j].pt.x, keypoints_last_ori_img_[j].pt.y, 1);
                        p_last_global = camera_intrinsic_matrix_inv * pixel_position_last * depth_img_last_.at<float>(keypoints_last_ori_img_[j].pt.y, keypoints_last_ori_img_[j].pt.x);
                        p_last_global = camera_orientation_matrix_last_ * p_last_global + camera_position_last_;

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
                bool iou_matched = false;
                for(int i = 0; i < bounding_boxes_last_frame_.size(); ++i){
                    double iou = bounding_boxes_curr_frame_[m].calculateIOU(bounding_boxes_last_frame_[i]);
                    
                    // If the bounding box is at the edges of the image, decrease the correction factor to make it easier to match.
                    double correction_factor = 1.0;
                    if(bounding_boxes_curr_frame_[m].x_min_ < 10 || bounding_boxes_curr_frame_[m].x_max_ > masks[0].cols - 10 || bounding_boxes_curr_frame_[m].y_min_ < 10 || bounding_boxes_curr_frame_[m].y_max_ > masks[0].rows - 10){
                        correction_factor = 0.5;
                    }
                    
                    if(iou > c_iou_threshold*correction_factor){
                        int id = track_ids_masks_last_[i];
                        iou_matched = true;
                        if(matched_track_ids.find(id) == matched_track_ids.end()){
                            if(id < 0){
                                id = next_tracking_id_;
                                next_tracking_id_ ++;
                                std::cout << "IOU Make it a new tracking id = " << id << std::endl;
                            }else{
                                std::cout << "IOU tracked object = " << id << std::endl;
                            }
                            track_ids_masks[m] = id;
                            matched_track_ids.insert(id);
                            break;
                        }
                    }
                }

                // If no matched tracking ID is found, create a new one.
                if(!iou_matched){
                    // If no matched tracking ID is found, create a new one.
                    int id = next_tracking_id_;
                    next_tracking_id_ ++;
                    track_ids_masks[m] = id;
                    std::cout << "No Keypoint and IOU matching. Make it a new tracking id = " << id << std::endl;
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

        // Set the copied message
        copied_msg.header.stamp = ros::Time::now();
        copied_msg.header.frame_id = "map";
        copied_msg.header.seq = seq_id_;
        
        for(size_t i = 0; i < masks.size(); ++i){
            copied_msg.objects[i].track_id = track_ids_masks[i];

            if(copied_msg.objects[i].label == "0"){
                copied_msg.objects[i].label = "Person";
            }else{
                copied_msg.objects[i].label = "Car";
            }
            std::cout << "Object label = " << copied_msg.objects[i].label << std::endl;
        }
        
        // Add a static semantic mask for the copied msg. The name of the mask is "classmmseg_"+ five digit number.
        if(update_with_semantic_seg){
            // cv::Mat semantic_mask = cv_bridge::toCvCopy(semantic_seg_img, sensor_msgs::image_encodings::BGR8)->image;
            
            cv_bridge::CvImagePtr cv_ptr;
            try{
                cv_ptr = cv_bridge::toCvCopy(semantic_seg_img);
            }
            catch (cv_bridge::Exception& e){
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }
            cv::Mat semantic_mask = cv_ptr->image;

            // Create a 8UC1 mask filled with labels
            cv::Mat semantic_mask_mono = cv::Mat::zeros(semantic_mask.rows, semantic_mask.cols, CV_8UC1);
            for(int i = 0; i < semantic_mask.rows; ++i){
                for(int j = 0; j < semantic_mask.cols; ++j){

                    cv::Vec3b color = semantic_mask.at<cv::Vec3b>(i, j);
                    // Check the label_id in g_label_color_map_default
                    int label_id = 0;
                    for(const auto &label_color : object_info_handler.label_color_map){
                        if(color == label_color.second){
                            label_id = label_color.first;
                            break;
                        }
                    }
                    semantic_mask_mono.at<uchar>(i, j) = label_id;
                }
            }

            // print object_info_handler.label_color_map
            // cv::imshow("semantic_mask", semantic_mask);
            // cv::imshow("semantic_mask_mono", semantic_mask_mono);
            // cv::waitKey(10);

            mask_kpts_msgs::MaskKpts mask_kpts_msg;
            mask_kpts_msg.track_id = 65535;
            mask_kpts_msg.label = "static";
            cv_bridge::CvImage mask_cv_image(std_msgs::Header(), "mono8", semantic_mask_mono);
            mask_kpts_msg.mask = *(mask_cv_image.toImageMsg());
            copied_msg.objects.push_back(mask_kpts_msg);
        }

        // Publish the message
        mask_pub_.publish(copied_msg);

        // Publish depth_img_curr_
        cv_bridge::CvImage cv_depth_image(copied_msg.header, "32FC1", depth_img_curr_);
        depth_image_repub_.publish(cv_depth_image.toImageMsg());

        // Publish camera pose
        geometry_msgs::PoseStamped camera_pose_msg;
        camera_pose_msg.header = copied_msg.header;
        camera_pose_msg.pose.position.x = camera_position_curr_[0];
        camera_pose_msg.pose.position.y = camera_position_curr_[1];
        camera_pose_msg.pose.position.z = camera_position_curr_[2];
        camera_pose_msg.pose.orientation.w = camera_orientation_curr_.w();
        camera_pose_msg.pose.orientation.x = camera_orientation_curr_.x();
        camera_pose_msg.pose.orientation.y = camera_orientation_curr_.y();
        camera_pose_msg.pose.orientation.z = camera_orientation_curr_.z();
        camera_pose_repub_.publish(camera_pose_msg);

        // Show the result in a single image
        showResult(masks, bounding_boxes_curr_frame_, track_ids_masks, copied_msg);

        // Update for the next iteration.
        tracking_ids_last_frame_ = tracking_ids_curr_frame_;
        matched_points_ready_ = false;
        bounding_boxes_last_frame_ = bounding_boxes_curr_frame_;
        track_ids_masks_last_ = track_ids_masks;

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

            // Find the center of the bounding box and add the track id
            int center_x = (bboxes[i].x_min_ + bboxes[i].x_max_) / 2;
            int center_y = (bboxes[i].y_min_ + bboxes[i].y_max_) / 2;
            std::string text = std::to_string(track_id);
            cv::putText(mask_image, text, cv::Point(center_x, center_y), cv::FONT_HERSHEY_SIMPLEX, 0.8, reversed_color, 1);
        }

        cv::imshow("mask_image", mask_image);
        cv::waitKey(10);
    }

    // The function is used to initialize the engine for superpoint and superglue
    void initEngine()
    {
        std::string package_path = ros::package::getPath("single_camera_tracking");
        std::string config_path = package_path + "/SuperPoint-SuperGlue-TensorRT/config/config.yaml";
        std::string model_dir = package_path + "/SuperPoint-SuperGlue-TensorRT/weights/";
        Configs configs(config_path, model_dir);

        height_ = configs.superglue_config.image_height;
        width_ = configs.superglue_config.image_width; 

        next_tracking_id_ = 1; //Start from 1. CHG
        matched_points_ready_ = false;

        seq_id_ = 0;

        camera_intrinsic_matrix << c_camera_fx, 0, c_camera_cx,
                                  0, c_camera_fy, c_camera_cy,
                                  0, 0, 1;
        camera_intrinsic_matrix_inv = camera_intrinsic_matrix.inverse().eval();

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
    }

    void run(){
        std::cout << "camera_pose_topic = " << camera_pose_topic << std::endl;
        std::cout << "rgb_image_topic = " << rgb_image_topic << std::endl;
        std::cout << "depth_image_topic = " << depth_image_topic << std::endl;

        // Subscribe to camera_rgb_image, camera_depth_image and camera pose synchronously
        message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub(nh_, rgb_image_topic, 1);
        message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, depth_image_topic, 1);
        message_filters::Subscriber<geometry_msgs::PoseStamped> camera_pose_sub(nh_, camera_pose_topic, 1);
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_image_sub, depth_image_sub, camera_pose_sub);
        sync.registerCallback(boost::bind(&TrackingNode::imageCallback, this, _1, _2, _3));

        segmentation_result_sub_ = nh_.subscribe("/mask_group", 1, &TrackingNode::instanceSegmentationResultCallback, this);

        image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/image_super_glued", 1);
        mask_pub_ = nh_.advertise<mask_kpts_msgs::MaskGroup>("/mask_group_super_glued", 1);
        key_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/key_points", 1);

        depth_image_repub_ = nh_.advertise<sensor_msgs::Image>("/camera/depth_repub", 1);
        camera_pose_repub_ = nh_.advertise<geometry_msgs::PoseStamped>("/camera/pose_repub", 1);

        original_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/original_point_cloud", 1);

        
        initEngine();

        ros::spin();
    }

    /// @brief The function is used for initializating the node
    void runWithSemanticImage(){        
        std::cout << "camera_pose_topic = " << camera_pose_topic << std::endl;
        std::cout << "rgb_image_topic = " << rgb_image_topic << std::endl;
        std::cout << "depth_image_topic = " << depth_image_topic << std::endl;

        // Subscribe to camera_rgb_image, camera_depth_image and camera pose synchronously
        message_filters::Subscriber<sensor_msgs::Image> rgb_image_sub(nh_, rgb_image_topic, 1);
        message_filters::Subscriber<sensor_msgs::Image> depth_image_sub(nh_, depth_image_topic, 1);
        message_filters::Subscriber<geometry_msgs::PoseStamped> camera_pose_sub(nh_, camera_pose_topic, 1);
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, geometry_msgs::PoseStamped> MySyncPolicy;
        message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), rgb_image_sub, depth_image_sub, camera_pose_sub);
        sync.registerCallback(boost::bind(&TrackingNode::imageCallback, this, _1, _2, _3));

        std::cout << "semantic_msg_available = true." << std::endl;
        message_filters::Subscriber<mask_kpts_msgs::MaskGroup> segmentation_result_sub(nh_, "/mask_group", 1);
        message_filters::Subscriber<sensor_msgs::Image> semantic_image_sub(nh_, "/semantic_image", 1);
        
        typedef message_filters::sync_policies::ApproximateTime<mask_kpts_msgs::MaskGroup, sensor_msgs::Image> MySyncPolicy2;
        message_filters::Synchronizer<MySyncPolicy2> sync2(MySyncPolicy2(10), segmentation_result_sub, semantic_image_sub);
        sync2.registerCallback(boost::bind(&TrackingNode::panopticSegmentationResultCallback, this, _1, _2));

        image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/image_super_glued", 1);
        mask_pub_ = nh_.advertise<mask_kpts_msgs::MaskGroup>("/mask_group_super_glued", 1);
        key_points_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/key_points", 1);

        depth_image_repub_ = nh_.advertise<sensor_msgs::Image>("/camera/depth_repub", 1);
        camera_pose_repub_ = nh_.advertise<geometry_msgs::PoseStamped>("/camera/pose_repub", 1);

        original_point_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/original_point_cloud", 1);

        
        initEngine();

        ros::spin();
    }


};

int main(int argc, char** argv){
    ros::init(argc, argv, "tracking_node");
    
    std::string setting_file = "coda.yaml";
    std::string object_info_csv_file = "object_info.csv";
    if(argc == 2){
        setting_file = argv[1];
    }else if(argc == 3){
        setting_file = argv[1];
        object_info_csv_file = argv[2];
    }else{
        std::cout << "Usage: rosrun single_camera_tracking tracking_node [setting_file] [object_info_csv_file]" << std::endl;
        return -1;
    }
    
    // Read parameters from yaml file
    std::string package_path = ros::package::getPath("single_camera_tracking");
    std::string config_path = package_path + "/cfg/" + setting_file;
    std::string object_info_csv_path = package_path + "/cfg/" + object_info_csv_file;
    object_info_handler.readObjectInfo(object_info_csv_path);


    YAML::Node config = YAML::LoadFile(config_path);
    c_camera_fx = config["camera_fx"].as<float>();
    c_camera_fy = config["camera_fy"].as<float>();
    c_camera_cx = config["camera_cx"].as<float>();
    c_camera_cy = config["camera_cy"].as<float>();
    points_too_far_threshold = config["points_too_far_threshold"].as<float>();
    points_too_close_threshold = config["points_too_close_threshold"].as<float>();
    c_vote_number_threshold = config["min_vote_number_threshold"].as<int>();
    c_iou_threshold = config["min_iou_threshold"].as<float>();
    c_bbox_size_threshold = config["min_bbox_size_threshold"].as<double>();
   
    rgb_image_topic = config["rgb_image_topic"].as<std::string>();
    depth_image_topic = config["depth_image_topic"].as<std::string>();
    camera_pose_topic = config["camera_pose_topic"].as<std::string>();

    bool semantic_msg_available = config["semantic_msg_available"].as<bool>();
    std::cout << "semantic_msg_available = " << semantic_msg_available << std::endl;

    std::cout << "c_camera_fx = " << c_camera_fx << ", " << "c_camera_fy = " << c_camera_fy << ", " << "c_camera_cx = " << c_camera_cx << ", " << "c_camera_cy = " << c_camera_cy << std::endl;
    std::cout << "points_too_far_threshold = " << points_too_far_threshold << ", " << "points_too_close_threshold = " << points_too_close_threshold << std::endl;
    std::cout << "c_vote_number_threshold = " << c_vote_number_threshold << ", " << "c_iou_threshold = " << c_iou_threshold << ", " << "c_bbox_size_threshold = " << c_bbox_size_threshold << std::endl;

    // Create a node
    TrackingNode tracking_node(semantic_msg_available);
    
    return 0;
}
