#include "super_glue.h"
#include "super_point.h"
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <ros/package.h>


int main(int argc, char** argv){

    ros::init(argc, argv, "superpointtest");

    std::string package_path = ros::package::getPath("single_camera_tracking");
    std::string config_path = package_path + "/SuperPoint-SuperGlue-TensorRT/config/test_config.yaml";
    std::string model_dir = package_path + "/SuperPoint-SuperGlue-TensorRT/weights/";
    Configs configs(config_path, model_dir);

    int img_height = configs.superglue_config.image_height;
    int img_width = configs.superglue_config.image_width; 

    // Create superpoint detector and superglue matcher. Build engine
    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()){
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return -1;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()){
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return -1;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;


    // Read image paths in a folder
    std::string image_folder ="/home/clarence/ros_ws/semantic_dsp_ws/src/Semantic_DSP_Map/data/VirtualKitti2/rgb/Camera_0";
    std::vector<std::string> image_paths;
    cv::glob(image_folder, image_paths);

    int image_num = image_paths.size();
    std::cout << "image_num: " << image_num << std::endl;

    int n = 0;
    while(ros::ok() && image_num > 1)
    {
        // Read n and n+1 images as a pair
        cv::Mat image0 = cv::imread(image_paths[n],cv::IMREAD_GRAYSCALE);
        cv::Mat image1 = cv::imread(image_paths[n+1],cv::IMREAD_GRAYSCALE);

        // Resize image
        cv::resize(image0, image0, cv::Size(img_width, img_height));
        cv::resize(image1, image1, cv::Size(img_width, img_height));

        // infer superpoint
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0, feature_points1;
        superpoint->infer(image0, feature_points0);
        superpoint->infer(image1, feature_points1);

        // infer superglue
        std::vector<cv::DMatch> supergluematches;
        superglue->matching_points(feature_points0, feature_points1, supergluematches);

        cv::Mat image0_show = cv::imread(image_paths[n]);
        cv::Mat image1_show = cv::imread(image_paths[n+1]);

        // Show superpoints of image 0
        for (int i = 0; i < feature_points0.cols(); i++)
        {
            cv::Point2f point;
            point.x = feature_points0(1, i) / img_width * image0_show.cols;
            point.y = feature_points0(2, i) / img_height * image0_show.rows;
            cv::circle(image0_show, point, 2, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("image0", image0_show);

        // Show superpoints of image 1
        for (int i = 0; i < feature_points1.cols(); i++)
        {
            cv::Point2f point;
            point.x = feature_points1(1, i) / img_width * image1_show.cols;
            point.y = feature_points1(2, i) / img_height * image1_show.rows;
            cv::circle(image1_show, point, 2, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("image1", image1_show);

        // Get keypoints of image 0 and image 1
        std::vector<cv::KeyPoint> keypoints_0, keypoints_1;
        for (int i = 0; i < feature_points0.cols(); i++)
        {
            cv::KeyPoint keypoint;
            keypoint.pt.x = feature_points0(1, i) / img_width * image0_show.cols;
            keypoint.pt.y = feature_points0(2, i) / img_height * image0_show.rows;
            keypoints_0.push_back(keypoint);
        }
        for (int i = 0; i < feature_points1.cols(); i++)
        {
            cv::KeyPoint keypoint;
            keypoint.pt.x = feature_points1(1, i) / img_width * image1_show.cols;
            keypoint.pt.y = feature_points1(2, i) / img_height * image1_show.rows;
            keypoints_1.push_back(keypoint);
        }

        // Draw matches and make the lines semi-transparent
        cv::Mat out_img;
        cv::drawMatches(image0_show, keypoints_0, image1_show, keypoints_1, supergluematches, out_img, cv::Scalar::all(-1),
                        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("matches", out_img);
        


        // If press 'q', exit. If press 'a' or 'A' show the previous image pair. If press 'd' or 'D' show the next image pair.
        char key = cv::waitKey(0);
        if (key == 'q' || key == 'Q' || key == 27){
            break;
        } else if (key == 'a' || key == 'A'){
            n--;
            if (n < 0){
                n = 0;
            }
        } else if (key == 'd' || key == 'D'){
            n++;
            if (n >= image_num - 1){
                n = image_num - 2;
            }
        }

    }

    cv::destroyAllWindows();

    return 0;
}