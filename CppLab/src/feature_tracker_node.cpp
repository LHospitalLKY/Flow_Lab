//
// Author: LHO LHospitalLKY@github.com 
//

#include "../util/line_tracking.h"
#include "feature_tracker.h"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>　
#include <sys/types.h> 
#include <opencv2/opencv.hpp>

/*
int main(int argc, char *argv[]) {

    // 打开两幅图
    // cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000000_10.png");
    // cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000000_11.png");

    cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638518077829376.png");
    cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638518227829504.png");

    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);

    std::vector<cv::Vec4f> prev_lines;
    std::vector<cv::Vec4f> cur_lines;
    std::vector<uchar> status;

    std::shared_ptr<LineOpticalFlow> lof = std::make_shared<LineOpticalFlow>();
    lof -> calc(frame_0_gray, frame_1_gray, prev_lines, cur_lines, status);
    // cv::Mat lines_show = lof -> drawTracking(frame_1, cur_lines);
    // cv::imshow("tracking lines", lines_show);

    // 画mask
    cv::Mat mask(frame_0.size(), CV_8UC1, cv::Scalar(255));
    for(int i = 0; i < cur_lines.size(); i++) {
        cv::Point2f start, end;
        start.x = cur_lines[i][0];
        start.y = cur_lines[i][1];
        end.x = cur_lines[i][2];
        end.y = cur_lines[i][3];
        cv::circle(mask, start, 5, 0, -1);
        cv::circle(mask, end, 5, 0, -1);
    }
    cv::imshow("mask", mask);
    cv::waitKey(0);

    // 提取新的线段
    std::vector<cv::Vec4f> new_lines_all;
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(LSD_REFINE_STD);
    lsd -> detect(frame_1_gray, new_lines_all);
    // 去掉在mask中的线段
    std::vector<cv::Vec4f> new_lines;
    for(int i = 0; i < new_lines_all.size(); i++) {
        cv::Point2f start, end;
        start.x = new_lines_all[i][0];
        start.y = new_lines_all[i][1];
        end.x = new_lines_all[i][2];
        end.y = new_lines_all[i][3];
        if(mask.at<uchar>(start.y, start.x) == 0) {
            continue;
        }
        if(mask.at<uchar>(end.y, end.x) == 0) {
            continue;
        }
        std::cout << "find new lines" << std::endl;
        new_lines.push_back(new_lines_all[i]);
    }

    cv::Mat new_lines_show(frame_1);
    if(new_lines.size() > 0) {
        std::cout << "new lines: " << new_lines.size() << std::endl;
        lsd -> drawSegments(new_lines_show, new_lines);
        cv::imshow("draw new lines", new_lines_show);
        cv::waitKey(0);
    } else {
        lsd -> drawSegments(new_lines_show, new_lines_all);
        cv::imshow("draw new lines", new_lines_show);
        std::cout << "not found new lines" << std::endl;
        cv::waitKey(0);
    }
    
    return 1;

}
*/

#define MAX_PATH_LEN (256)

using namespace std;

void getFiles(std::string path, std::vector<std::string> &files);

int main(int argc, char *argv[]) {

    std::string data_root = "/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/TUM/f-xyz1/1/rgbd_dataset_freiburg1_xyz/rgb";
    vector<std::string> files;
    
    // 读取csv文件
    std::ifstream csv;
    csv.open("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/TUM/f-xyz1/1/rgbd_dataset_freiburg1_xyz/rgb.txt");
    std::string csv_line;
    int k = 1;
    while(std::getline(csv, csv_line)) {
        std::cout << csv_line << std::endl;
        std::istringstream sin(csv_line);
        vector<std::string> fields;
        std::string field;
        while(getline(sin, field, '/')) {
            fields.push_back(field);
        }
        if(k == 1) {
            k++;
            continue;
        }
        double time = std::atof(fields[0].c_str());
        // EuRoC
        // std::string file_name = fields[1].substr(0, fields[1].length() - 1);
        std::string file_name = fields[1].substr(0, fields[1].length());

        std::cout << "Time Stamp: " << time << std::endl;
        std::cout << file_name << std::endl;

        files.push_back(file_name);
    }

    std::cout << "------------------" << std::endl;

    std::shared_ptr<FeatureTracker> feature_tracker = std::make_shared<FeatureTracker>();

    // 保存到视频中
    int fourcc = CV_FOURCC('M', 'J', 'P', 'G');
    cv::VideoWriter drawLines_writer, flow_writer;
    drawLines_writer.open("tum_tracking.mp4", fourcc, 20, cv::Size(640, 480));
    flow_writer.open("tum_flow.mp4", fourcc, 20, cv::Size(640, 480));

    for(int i = 1; i < files.size(); i++) {
        std::string name_0 = files[i - 1];
        std::string name_1 = files[i];
        cv::Mat frame_0, frame_1, frame_0_gray, frame_1_gray;
        // EuRoc
        // frame_0 = cv::imread(data_root + "/" + files[i - 1], CV_8UC1);
        // frame_1 = cv::imread(data_root + "/" + files[i], CV_8UC1);
        frame_0 = cv::imread(data_root + "/" + files[i - 1]);
        frame_1 = cv::imread(data_root + "/" + files[i]);
        cv::imshow("frame_0", frame_0);
        cv::imshow("frame_1", frame_1);

        cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
        cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);

        // 去畸变
        Eigen::Matrix3d K_eigen;
        /*
        K_eigen << 4.616e+02, 0, 3.630e+02, 
                    0, 4.603e+02, 2.481e+02, 
                    0, 0, 1;*/
        K_eigen << 1, 0, 0, 
                    0, 1, 0, 
                    0, 0, 1;
        cv::Mat K;
        cv::eigen2cv(K_eigen, K);
        std::vector<double> dist(4);
        /*
        dist[0] = -2.917e-01;
        dist[1] = 8.228e-02;
        dist[2] = 5.333e-05;
        dist[3] = -1.578e-04;*/
        dist[0] = 0;
        dist[1] = 0;
        dist[2] = 0;
        dist[3] = 0;
        // dis
        // cv::Mat map1, map2;
        cv::Mat drawLines, flow_color;
        feature_tracker -> readIntrinsicParameter(K, dist);
        feature_tracker -> readImage(frame_0_gray, 0, drawLines, flow_color);
        drawLines_writer.write(drawLines);
        flow_writer.write(flow_color);
    }

    // drawLines_writer.close();
    // flow_writer.close();
}

void getFiles(std::string path, std::vector<std::string> &files) {
    DIR *d = NULL;
    struct dirent *dp = NULL; /* readdir函数的返回值就存放在这个结构体中 */
    struct stat st;    
    char p[MAX_PATH_LEN] = {0};
    
    // 如果path路径不对
    if(stat(path.c_str(), &st) < 0 || !S_ISDIR(st.st_mode)) {
        std::cout << "invalid path: " << path << std::endl;
        return;
    }
    // 打开路径失败
    if(!(d = opendir(path.c_str()))) {
        std::cout << "opendir[" << path << "] error: %m" << std::endl;
        return;
    }

    while((dp = readdir(d)) != NULL) {
        /* 把当前目录.，上一级目录..及隐藏文件都去掉，避免死循环遍历目录 */
        if((!strncmp(dp->d_name, ".", 1)) || (!strncmp(dp->d_name, "..", 2)))
            continue;

        snprintf(p, sizeof(p) - 1, "%s, /%s", path, dp->d_name);
        stat(p, &st);
        /*
        if(!S_ISDIR(st.st_mode)) {
            printf("%s\n", dp->d_name);
        } else {
            printf("%s/\n", dp->d_name);
            getFiles(p);
        }
        */
        files.push_back(std::string(dp -> d_name));

    }
    closedir(d);

    return;
}