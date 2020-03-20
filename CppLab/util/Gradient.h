#ifndef __GRADIENT_h__
#define __GRADIENT_h__

#ifdef __GRADIENT_h__GLOBAL
    #define __GRADIENT_h__EXTERN 
#else
    #define __GRADIENT_h__EXTERN extern
#endif

// 求解一幅图像x方向的梯度、y方向的梯度、梯度方向角、梯度模长

#include <iostream>
#include <memory>
#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>

typedef struct GradientMap {
    cv::Mat dx;
    cv::Mat dy;
    cv::Mat angle;
    cv::Mat magnitude;
    cv::Mat magnitudeLevel;
};

class Gradient {
public:
    Gradient();

    // 计算
    void compute(const cv::Mat &img);

    // 返回结构体
    GradientMap& returnGradient();

private:
    GradientMap gm;

};



#endif // __GRADIENT_h__
