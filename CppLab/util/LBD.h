#ifndef __LBD_h__
#define __LBD_h__

#ifdef __LBD_h__GLOBAL
    #define __LBD_h__EXTERN 
#else
    #define __LBD_h__EXTERN extern
#endif

#include "Gradient.h"

#include <iostream>
#include <vector>
#include <memory>

#define EIGEN_USE_BLAS
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/StdVector>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>

// #define Debug

// Band结构体
struct Band {
    cv::Mat gx;
    cv::Mat gy;
    cv::Point2f center;
    // weight的顺序为index从小到达，无论正负
    std::vector<float> global_weight;
    std::vector<float> local_weight;
};

typedef Eigen::Matrix<double, 4, 3> Matrix43;
typedef Eigen::Matrix<double, 8, 5> MatrixBD;

class LSR {
public:
    LSR();
    // 使用Vec4f表示的线段求解线特征描述
    void computeBands(const cv::Mat &img, const cv::Vec4f &keyline, const GradientMap &gm);
    // 使用KeyLine格式的线特征来求解描述
    void computeBands(const cv::Mat &img, const cv::line_descriptor::KeyLine &keyline, const GradientMap &gm);
    void showGradient();
    
    MatrixBD& returnDescriptor();

private:
    void computeLocalCenter();
    void computeGradient(std::vector<cv::Point2f> &line_pixels, const GradientMap &gm);
    MatrixBD computeDescriptor();
    Matrix43 computeDescriptor_singleBand(Band &band);

private:
    // 五条band
    Band band1_, band2_, band3_, band4_, band5_;
    // 全局中点
    cv::Point2f global_center_;
    // 线段端点
    cv::Point2f start_point_, end_point_;
    // 线段长度
    float length_;
    // 线段方向
    Eigen::Vector2d d_l_, d_orth_;

    // 必要的参数
    int m;
    int w;
    double theta_g;
    double theta_l;

    // 描述子
    MatrixBD LBD_;
};


#endif // __LBD_h__

