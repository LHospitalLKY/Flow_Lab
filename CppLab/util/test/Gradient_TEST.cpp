//
// Author: LHO LHospitalLKY@github.com 
//

#include "../Gradient.h"

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/line_descriptor.hpp>

int main(int argc, char *argv[]) {

    GradientMap gm;
    std::shared_ptr<Gradient> gradient = std::make_shared<Gradient>();

    cv::Mat frame_0 = cv::imread("/home/lho/SLAM/FlowNet/Flow_lab/Line Segment Detector/data/LK1.png", CV_8UC1);

    auto start_time = std::chrono::system_clock::now();
    gradient -> compute(frame_0);
    gm = gradient -> returnGradient();
    auto end_time = std::chrono::system_clock::now();

    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "My time uesd: " << time_used.count() << "ms" << std::endl;

    cv::imshow("gradient_level", gm.magnitudeLevel);
    cv::waitKey(0);

    return 1;

}