#ifndef __LINE_TRACKING_h__
#define __LINE_TRACKING_h__

#ifdef __LINE_TRACKING_h__GLOBAL
    #define __LINE_TRACKING_h__EXTERN 
#else
    #define __LINE_TRACKING_h__EXTERN extern
#endif

#include <iostream>

#include "show_flow.h"
#include "read_write.h"

#include <chrono>
#include <omp.h>

#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

struct Duration {
    int64_t all_duration;
    int64_t flow_duration;
    int64_t line_duration;
};

class LineOpticalFlow {
public:
    LineOpticalFlow();

    // calc
    void calc(
         cv::Mat &prev,
         cv::Mat &cur, 
         std::vector<cv::Vec4f> &prev_lines, 
         std::vector<cv::Vec4f> &cur_lines, 
         std::vector<int> &status
        );

    // 画图, 画出预测得到的直线与输入图像的合成图
    cv::Mat drawTracking(cv::Mat &cur, std::vector<cv::Vec4f> &cur_lines);

    // 返回时间
    void returnDuration(Duration &durations);

private:
    bool inBoundBox(double x, double y, double x_max, double y_max);

private:
    // std::vector<cv::Vec4f> &prev_lines_;
    // std::vector<cv::Vec4f> &cur_lines_;
    // std::vector<int> &status_;

    // deepflow光流
    cv::Ptr<cv::DenseOpticalFlow> deepflow_;
    // 直线特征检测子
    cv::Ptr<cv::LineSegmentDetector> lsd_;

    int64_t all_time_duration_;
    int64_t flowCalc_time_duration_;
    int64_t lineSeg_time_duration_;
};


#endif // __LINE_TRACKING_h__
