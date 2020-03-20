#ifndef __SHOWFLOW_h__
#define __SHOWFLOW_h__

#ifdef __SHOWFLOW_h__GLOBAL
    #define __SHOWFLOW_h__EXTERN 
#else
    #define __SHOWFLOW_h__EXTERN extern
#endif

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
 
#define UNKNOWN_FLOW_THRESH 1e9
#define UNKNOWN_FLOW_THRESH_MIN 1e-10

void makecolorwheel(vector<Scalar> &colorwheel);
void motionToColor(Mat flow, Mat &color);


#endif // __SHOWFLOW_h__
