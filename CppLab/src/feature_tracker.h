#ifndef __HEATURE_TRACKER_h__
#define __HEATURE_TRACKER_h__

#ifdef __HEATURE_TRACKER_h__GLOBAL
    #define __HEATURE_TRACKER_h__EXTERN 
#else
    #define __HEATURE_TRACKER_h__EXTERN extern
#endif

#define EIGEN_USE_BLAS

#include "../util/line_tracking.h"

#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;

// bool inBorder(const cv::Vec4f &line);

// TODO: 考虑如何写reduceVector系列函数
void reduceVector(vector<cv::Vec4f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

// feature tracking类
class FeatureTracker {
public:
    FeatureTracker();

    void readImage(const cv::Mat &img, double cur_time, cv::Mat &drawLines, cv::Mat &flow);

    void setMask();

    void addLines();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const cv::Mat &K, const std::vector<double> &dist);

    void showUndistortion();

    void rejectWithF();

    void undistortedPoints();

private:
    bool inBorder(const cv::Vec4f &line);

    void lineDetect(const cv::Mat &img, vector<cv::Vec4f> &lines, const cv::Mat &mask);

private:
    cv::Mat mask_;
    cv::Mat fisheye_mask_;
    cv::Mat prev_img_, cur_img_, forw_img_;
    vector<cv::Vec4f> n_lines_;
    vector<cv::Vec4f> prev_lines_, cur_lines_, forw_lines_;
    vector<cv::Vec4f> prev_un_lines_, cur_un_lines_;
    
    vector<int> ind_;
    vector<int> track_cnt_;
    map<int, cv::Vec4f> cur_un_lines_map_;
    map<int, cv::Vec4f> prev_un_lines_map_;

    cv::Mat K_;
    std::vector<double> dist_;

    double cur_time_;
    double prev_time_;

    static int n_id_;
};


#endif // __HEATURE_TRACKER_h__
