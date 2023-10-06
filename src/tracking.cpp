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

class TrackingNode{
public:
    TrackingNode(){
        initialization();
    };
    ~TrackingNode(){};

private:
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher image_pub_;

    std::shared_ptr<SuperPoint> superpoint_;
    std::shared_ptr<SuperGlue> superglue_;

    cv::Mat img_last_;
    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points_last_;
    bool last_frame_ready_ = false;

    int width_, height_;


    /// @brief The function is used to get superpoint and superglue inference result
    /// @param img 
    void superglueInference(cv::Mat &img){
        // Resize image
        cv::resize(img, img, cv::Size(width_, height_));

        // Add a timer
        auto start = std::chrono::high_resolution_clock::now();

        // infer superpoint
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points;
        superpoint_->infer(img, feature_points);

        if(!last_frame_ready_){
            last_frame_ready_ = true;
            img_last_ = img;
            feature_points_last_ = feature_points;
            return;
        }

        // infer superglue
        std::vector<cv::DMatch> superglue_matches;
        superglue_->matching_points(feature_points_last_, feature_points, superglue_matches);

        // End of timer
        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        std::cout << "Inference cost " << duration / 1000 << " MS" << std::endl;

        // draw matches
        cv::Mat match_image;
        std::vector<cv::KeyPoint> keypoints0, keypoints1;
        for(size_t i = 0; i < feature_points_last_.cols(); ++i){
            double score = feature_points_last_(0, i);
            double x = feature_points_last_(1, i);
            double y = feature_points_last_(2, i);
            keypoints0.emplace_back(x, y, 8, -1, score);
        }
        for(size_t i = 0; i < feature_points.cols(); ++i){
            double score = feature_points(0, i);
            double x = feature_points(1, i);
            double y = feature_points(2, i);
            keypoints1.emplace_back(x, y, 8, -1, score);
        }

        cv::drawMatches(img_last_, keypoints0, img, keypoints1, superglue_matches, match_image);
       
        // Publish image
        cv_bridge::CvImage cv_image;
        cv_image.image = match_image;
        cv_image.encoding = "bgr8";
        image_pub_.publish(cv_image.toImageMsg());

        // Save image and feature points
        cv::imshow("match_image", match_image);
        cv::waitKey(10);

        img_last_ = img;
        feature_points_last_ = feature_points;
    }


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

        cv::Mat img = cv_ptr->image;
        // Convert to grayscale
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
        superglueInference(img);
    }


    /// @brief The function is used for initializating the node
    void initialization(){
        image_sub_ = nh_.subscribe("/camera_rgb_image", 1, &TrackingNode::imageCallback, this);
        image_pub_ = nh_.advertise<sensor_msgs::Image>("/camera/image_super_glued", 1);

        std::string package_path = ros::package::getPath("single_camera_tracking");
        std::string config_path = package_path + "/SuperPoint-SuperGlue-TensorRT/config/config.yaml";
        std::string model_dir = package_path + "/SuperPoint-SuperGlue-TensorRT/weights/";
        Configs configs(config_path, model_dir);

        height_ = configs.superglue_config.image_height;
        width_ = configs.superglue_config.image_width;

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
