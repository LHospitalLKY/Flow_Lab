#ifndef __EVALUATION_h__
#define __EVALUATION_h__

#ifdef __EVALUATION_h__GLOBAL
    #define __EVALUATION_h__EXTERN 
#else
    #define __EVALUATION_h__EXTERN extern
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>

double endPointError(cv::Mat &estimFlow, cv::Mat &gtFlow, cv::Mat &EPE_Mat);
double angleError(cv::Mat &estimFlow, cv::Mat &gtFlow, cv::Mat &AE_Mat); 

#endif // __EVALUATION_h__
