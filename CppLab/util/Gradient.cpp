//
// Author: LHO LHospitalLKY@github.com 
//

#include "Gradient.h"

Gradient::Gradient() {}

void Gradient::compute(const cv::Mat &img) {
    // 判断img是不是CV_8UC1格式
    assert(img.type() == CV_8UC1);

    // 计算梯度
    int height = img.size().height;
    int width = img.size().width;

    gm.dx = cv::Mat::zeros(img.size(), CV_32FC1);
    gm.dy = cv::Mat::zeros(img.size(), CV_32FC1);
    gm.angle = cv::Mat::zeros(img.size(), CV_32FC1);
    gm.magnitude = cv::Mat::zeros(img.size(), CV_32FC1);
    gm.magnitudeLevel = cv::Mat::zeros(img.size(), CV_8UC1);

#pragma omp parallel for
    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            if(i == height || j == width) {
                gm.dx.at<float>(i, j) = 0;
                gm.dy.at<float>(i, j) = 0;
                gm.magnitude.at<float>(i, j) = 0;
                gm.angle.at<float>(i, j) = 0;
            }
            // std::cout << j << std::endl;

            float I1, I2, I3, I4;
            I1 = 0.5*((float)img.at<uchar>(i + 1, j + 1) + (float)img.at<uchar>(i + 1, j) - (float)img.at<uchar>(i, j + 1) - (float)img.at<uchar>(i , j));
            I2 = 0.5*((float)img.at<uchar>(i + 1, j + 1) + (float)img.at<uchar>(i, j + 1) - (float)img.at<uchar>(i + 1, j) - (float)img.at<uchar>(i , j));

            I3 = std::sqrt(I1*I1 + I2*I2);
            I4 = std::atan2(I1, -I2);
            
            gm.dx.at<float>(i, j) = I1;
            gm.dy.at<float>(i, j) = I2;
            gm.magnitude.at<float>(i, j) = I3;
            gm.magnitudeLevel.at<uchar>(i, j) = (int)I3;
            gm.angle.at<float>(i, j) = I4;
        }
    }
}

GradientMap& Gradient::returnGradient() {
    return gm;
}