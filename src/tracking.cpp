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
    bool matched_points_ready_ = false;

    int width_, height_; // image size
    
    std::vector<cv::KeyPoint> keypoints_last_ori_img_; // keypoints of last frame in real-size image coordinate
    std::vector<cv::KeyPoint> keypoints_current_ori_img_; // keypoints of this frame in real-size image coordinate
    std::vector<cv::DMatch> superglue_matches_; // superglue matches

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

        matched_points_ready_ = true;

        // End of timer
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference cost " << duration / 1000 << " MS" << std::endl;
    }


    /// @brief The function is used to get superpoint and superglue inference result
    /// @param img 
    void superglueInference(cv::Mat &img, bool draw_match = true){
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
            keypoints_last_ori_img_ = keypoints_current_ori_img_;
            keypoints_last = keypoints_current;
            return;
        }

        // infer superglue
        std::vector<cv::DMatch> superglue_matches;
        superglue_->matching_points(feature_points_last, feature_points_current, superglue_matches);
        superglue_matches_ = superglue_matches;

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
        keypoints_last_ori_img_ = keypoints_current_ori_img_;
        keypoints_last = keypoints_current;
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
        
        // Iterate all matches in superglue_matches_. Find the matches whose current keypoint is in the mask
        for(const auto &match : superglue_matches_)
        {
            // Get the current keypoint and matched last keypoint
            cv::KeyPoint current_keypoint = keypoints_current_ori_img_[match.trainIdx];
            cv::KeyPoint matched_last_keypoint = keypoints_last_ori_img_[match.queryIdx];
            // Check which mask the current keypoint is in
            for(size_t i = 0; i < masks.size(); ++i){
                if(masks[i].at<uchar>(current_keypoint.pt.y, current_keypoint.pt.x) == 1){
                    // The current keypoint is in the mask. Add it to the copied mask's keypoints.
                    single_camera_tracking::Keypoint kpt_this, kpt_last;
                    kpt_this.x = current_keypoint.pt.x;
                    kpt_this.y = current_keypoint.pt.y;
                    kpt_last.x = matched_last_keypoint.pt.x;
                    kpt_last.y = matched_last_keypoint.pt.y;
                    copied_msg.objects[i].kpts_curr.push_back(kpt_this);
                    copied_msg.objects[i].kpts_last.push_back(kpt_last);

                    break;
                }
            }
        }

        // Publish the copied message
        copied_msg.header.stamp = ros::Time::now();
        mask_pub_.publish(copied_msg);

        // Show the masks and matched current keypoints in one image
        cv::Mat mask_image = cv::Mat::zeros(masks[0].rows, masks[0].cols, CV_8UC3);
        for(size_t i = 0; i < masks.size(); ++i){
            cv::Mat mask = masks[i];
            cv::Mat mask_color = cv::Mat::zeros(mask.rows, mask.cols, CV_8UC3);

            // Set a random color for each mask
            cv::RNG rng(i);
            cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            mask_color.setTo(color, mask);
            mask_image = mask_image + mask_color;

            // Add matched current keypoints
            cv::Scalar reversed_color = cv::Scalar(255 - color[0], 255 - color[1], 255 - color[2]);
            for(const auto &kpt : copied_msg.objects[i].kpts_curr){
                cv::circle(mask_image, cv::Point(kpt.x, kpt.y), 2, reversed_color, -1);
            }
        }
        cv::imshow("mask_image", mask_image);
        cv::waitKey(10);

        matched_points_ready_ = false;
        return;
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
