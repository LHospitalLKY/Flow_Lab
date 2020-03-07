#ifndef __READANDWRITE_h__
#define __READANDWRITE_h__

#ifdef __READANDWRITE_h__GLOBAL
    #define __READANDWRITE_h__EXTERN 
#else
    #define __READANDWRITE_h__EXTERN extern
#endif

// 读取和写入.flo文件

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

#define TAR 202021.250

void write(std::string fileName, cv::Mat &flo);
void read(std::string fileName, cv::Mat &flo);


#endif // __READANDWRITE_h__
