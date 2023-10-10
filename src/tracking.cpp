//
// Created by haoyuefan on 2021/9/22.
//

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
#include <single_camera_tracking/MaskGroup.h>
#include <single_camera_tracking/MaskKpts.h>
#include <single_camera_tracking/Keypoint.h>
#include <unordered_set>


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
    ros::Subscriber raw_image_sub_, segmentation_result_sub_;
    ros::Publisher image_pub_, mask_pub_;

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperGlue> superglue_;

    cv::Mat img_last_;
    bool matched_points_ready_;

    int width_, height_; // image size
    
    std::vector<cv::KeyPoint> keypoints_last_ori_img_; // keypoints of last frame in real-size image coordinate
    std::vector<cv::KeyPoint> keypoints_current_ori_img_; // keypoints of this frame in real-size image coordinate
    
    std::vector<int> tracking_ids_last_frame_; // track id of keypoints in the last frame
    std::vector<int> tracking_ids_current_frame_; // track id of keypoints in this frame

    std::vector<int> track_ids_masks_last_; // track id of masks in the last frame

    std::vector<BoundingBox> bounding_boxes_last_frame_; // bounding boxes of objects in the last frame
    std::vector<BoundingBox> bounding_boxes_current_frame_; // bounding boxes of objects in this frame
    
    int next_tracking_id_; // next track id

    int c_min_kpts_to_track_; // minimum number of keypoints to track
    double bbox_size_threshold_; // minimum size of bounding box to track

    std::vector<cv::DMatch> superglue_matches_; // superglue matches

    // define a color map to show different track ids
    std::vector<cv::Scalar> color_map_;

    /// @brief Image callback function
    /// @param msg 
    void imageCallback(const sensor_msgs::ImageConstPtr& msg){
        cv_bridge::CvImagePtr cv_ptr;
        try{
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
        }
        catch (cv_bridge::Exception& e){
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Add a timer
        auto start = std::chrono::high_resolution_clock::now();

        cv::Mat img = cv_ptr->image;
        // Convert to grayscale
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
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points_current;
        static Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points_last;
        superpoint_->infer(img, feature_points_current);

        // Find keypoint correspondences
        static std::vector<cv::KeyPoint> keypoints_last;
        std::vector<cv::KeyPoint> keypoints_current;
        keypoints_current_ori_img_.clear();
        for(size_t i = 0; i < feature_points_current.cols(); ++i){
            double score = feature_points_current(0, i);
            double x = feature_points_current(1, i);
            double y = feature_points_current(2, i);
            keypoints_current.emplace_back(x, y, 8, -1, score);

            // Convert to real-size image coordinate
            x = x / width_ * img_ori_width;
            y = y / height_ * img_ori_height;
            keypoints_current_ori_img_.emplace_back(x, y, 8, -1, score);
        }

        static bool last_frame_ready = false;
        if(!last_frame_ready){
            last_frame_ready = true;
            img_last_ = img;
            feature_points_last = feature_points_current;
            keypoints_last = keypoints_current;
            return;
        }

        // infer superglue
        std::vector<cv::DMatch> superglue_matches;
        superglue_->matching_points(feature_points_last, feature_points_current, superglue_matches);
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
            cv::drawMatches(img_last_, keypoints_last, img, keypoints_current, superglue_matches, match_image);

            // Publish image
            cv_bridge::CvImage cv_image;
            cv_image.image = match_image;
            cv_image.encoding = "bgr8";
            image_pub_.publish(cv_image.toImageMsg());

            // Save image and feature points
            cv::imshow("match_image", match_image);
            cv::waitKey(10);
        }
       
        img_last_ = img;
        feature_points_last = feature_points_current;
        keypoints_last = keypoints_current;
        matched_points_ready_ = true;
    }

    /// @brief The function is used to callback segmentation result
    void segmentationResultCallback(const single_camera_tracking::MaskGroup& msg){
        if(!matched_points_ready_){
            std::cout << "No matched points ready !!!!!!!!!" << std::endl;
            return;
        }

        if(msg.objects.size() == 0){
            std::cout << "No mask received !!!!!!!!!" << std::endl;
            matched_points_ready_ = false;
            return;
        }

        // Copy the message to a local variable
        single_camera_tracking::MaskGroup copied_msg = msg;

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
        bounding_boxes_current_frame_.clear();
        for(size_t i = 0; i < msg.objects.size(); ++i){
            BoundingBox bounding_box;
            bounding_box.setByTLAndBR(msg.objects[i].bbox_tl.x, msg.objects[i].bbox_tl.y, msg.objects[i].bbox_br.x, msg.objects[i].bbox_br.y);
            bounding_boxes_current_frame_.emplace_back(bounding_box);
        }

        // Iterate all masks. Find the keypoints in each mask and store their indices.
        std::vector<std::vector<int>> keypoints_in_masks(masks.size());
        for (int i = 0; i < keypoints_current_ori_img_.size(); i++) {
            const auto& kp = keypoints_current_ori_img_[i];
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

        // Initialize tracking_ids_current_frame_ with -1
        tracking_ids_current_frame_ = std::vector<int>(keypoints_current_ori_img_.size(), -1);
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
        cv::Mat match_matrix = cv::Mat::zeros(keypoints_current_ori_img_.size(), keypoints_last_ori_img_.size(), CV_8U);
        for(const cv::DMatch& match : superglue_matches_) {
            match_matrix.at<uchar>(match.trainIdx, match.queryIdx) = 1;
        }

        // Iterate all masks. Find the keypoints in each mask and store their indices.
        std::unordered_set<int> matched_track_ids;
        for (int m = 0; m < masks.size(); m++) {
            std::cout << "mask " << m << " total mask size = " << masks.size() << std::endl;
            
            //Ignore the mask if it is too small.
            if(bounding_boxes_current_frame_[m].size() < bbox_size_threshold_){
                std::cout << "mask " << m << " is too small. Ignore it." << std::endl;
                track_ids_masks[m] = -1;
                continue;
            }

            // Vote for the tracking ID with the most matched keypoints.
            std::map<int, int> id_votes;
            for (int i : keypoints_in_masks[m]) {
                for (int j = 0; j < keypoints_last_ori_img_.size(); j++) {
                    if (match_matrix.at<uchar>(i, j) > 0) { // If the current keypoint is matched to a last keypoint.
                        int tracking_id = tracking_ids_last_frame_[j];
                        if (tracking_id >= 0) {
                            id_votes[tracking_id]++;
                        }

                        // Storing matched keypoints for each mask. To be published later.
                        single_camera_tracking::Keypoint kpt_curr, kpt_last;
                        kpt_curr.x = keypoints_current_ori_img_[i].pt.x;
                        kpt_curr.y = keypoints_current_ori_img_[i].pt.y;
                        kpt_last.x = keypoints_last_ori_img_[j].pt.x;
                        kpt_last.y = keypoints_last_ori_img_[j].pt.y;
                        copied_msg.objects[m].kpts_curr.push_back(kpt_curr);
                        copied_msg.objects[m].kpts_last.push_back(kpt_last);

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
            if (best_tracking_id >=0 && best_votes >= 3 && matched_track_ids.find(best_tracking_id) == matched_track_ids.end()) {
                track_ids_masks[m] = best_tracking_id;
                matched_track_ids.insert(best_tracking_id);
                std::cout << "tracked object = " << best_tracking_id << std::endl;
            } 
            else if(keypoints_in_masks[m].size() >= c_min_kpts_to_track_) { // If the number of keypoints in the mask is large enough but no matched found, use a new tracking ID.
                track_ids_masks[m] = next_tracking_id_;
                next_tracking_id_ ++;
                std::cout << "Use next_tracking_id_ = " << next_tracking_id_ << std::endl;
            } 
            else{
                // Try to match by IOU with the bounding boxes of the last frame in case too less features are matched.
                for(int i = 0; i < bounding_boxes_last_frame_.size(); ++i){
                    double iou = bounding_boxes_current_frame_[m].calculateIOU(bounding_boxes_last_frame_[i]);
                    if(iou > 0.5){
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
                tracking_ids_current_frame_[i] = track_ids_masks[m];
            }
        }

        // Publish the copied message
        copied_msg.header.stamp = ros::Time::now();
        for(size_t i = 0; i < masks.size(); ++i){
            copied_msg.objects[i].track_id = track_ids_masks[i];
        }
        mask_pub_.publish(copied_msg);

        // Show the result in a single image
        showResult(masks, bounding_boxes_current_frame_, track_ids_masks, copied_msg);

        // Update for the next iteration.
        tracking_ids_last_frame_ = tracking_ids_current_frame_;
        matched_points_ready_ = false;
        bounding_boxes_last_frame_ = bounding_boxes_current_frame_;
        track_ids_masks_last_ = track_ids_masks;

        std::cout << "segmentationResultCallback finished" << std::endl;
        return;
    }

    /// @brief The function is used to show the tracking result in a single image
    void showResult(const std::vector<cv::Mat> &masks, const std::vector<BoundingBox> &bboxes, const std::vector<int> &track_ids_masks, const single_camera_tracking::MaskGroup &msg)
    {
        // Show the masks and matched current keypoints in one image
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

            // Add matched current keypoints
            cv::Scalar reversed_color = cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]);
            for(const auto &kpt : msg.objects[i].kpts_curr){
                cv::circle(mask_image, cv::Point(kpt.x, kpt.y), 2, reversed_color, -1);
            }

            // Find the center of the matched current keypoints and write the track id
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
        raw_image_sub_ = nh_.subscribe("/camera_rgb_image", 1, &TrackingNode::imageCallback, this);
        segmentation_result_sub_ = nh_.subscribe("/mask_group", 1, &TrackingNode::segmentationResultCallback, this);

        image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/image_super_glued", 1);
        mask_pub_ = nh_.advertise<single_camera_tracking::MaskGroup>("/mask_group_super_glued", 1);

        std::string package_path = ros::package::getPath("single_camera_tracking");
        std::string config_path = package_path + "/SuperPoint-SuperGlue-TensorRT/config/config.yaml";
        std::string model_dir = package_path + "/SuperPoint-SuperGlue-TensorRT/weights/";
        Configs configs(config_path, model_dir);

        height_ = configs.superglue_config.image_height; //240
        width_ = configs.superglue_config.image_width; //320

        next_tracking_id_ = 0;
        matched_points_ready_ = false;

        c_min_kpts_to_track_ = 5;
        bbox_size_threshold_ = 3000;

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
    TrackingNode tracking_node;

    return 0;
}
