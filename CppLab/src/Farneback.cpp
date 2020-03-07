//
// Author: LHO LHospitalLKY@github.com 
//

#include <iostream>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#include <opencv2/opencv.hpp>
#include <opencv2/optflow.hpp>

#include "../util/show_flow.h"
#include "../util/read_write.h"

#define MAX_PATH_LEN (256)

using namespace std;

// 获取子文件夹或子文件图像文件名
void getFiles(std::string path, std::vector<std::string> &files);
// MPI training clean数据的定量实验
void MPITrainingClean();

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



int main(int argc, char *argv[]) {

    // MPI traning clean数据集


    // getFiles(path, files);

    MPITrainingClean();

    return 1;
}

void MPITrainingClean() {
    std::string root_path = "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/MPI-Sintel-complete/training";
    // 测试数据集路径
    std::string test_path = "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/MPI-Sintel-complete/training/clean";
    // ground truth路径
    std::string ground_truth_path = "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/MPI-Sintel-complete/training/flow";

    std::vector<std::string> sub_folders_test;
    std::vector<std::string> sub_folders_gt;

    getFiles(test_path, sub_folders_test);
    getFiles(ground_truth_path, sub_folders_gt);

    for(int i = 0; i < sub_folders_gt.size(); i++) {
        // 找到sub_folders_test中对应文件夹名的位置
        std::vector<std::string>::iterator it = std::find(sub_folders_test.begin(), sub_folders_test.end(), sub_folders_gt[i]);

        if(it != sub_folders_test.end()) {
            std::cout << "Operate in sequence: " << *it << std::endl;
            std::string sequence_path = test_path + "/" + *it;
            std::string gt_sequence_path = ground_truth_path + "/" + *it;
            std::vector<std::string> imgsName;
            std::vector<std::string> gtflowsName;
            getFiles(sequence_path, imgsName);
            getFiles(gt_sequence_path, gtflowsName);
            // 将数据读入队列中
            std::vector<cv::Mat> imgsQueue;
            std::vector<cv::Mat> gtsQueue;
            if(imgsName.size() > 0) {
                for(int k = 0; k < imgsName.size(); k++) {
                    imgsQueue.push_back(
                        cv::imread(sequence_path + "/" + imgsName[k], CV_8UC1)
                    );
                }
            }
            // 读入groundtruth
            for(int k = 0; k < gtflowsName.size(); k++) {
                cv::Mat gt_flow;
                read(gt_sequence_path + "/" + gtflowsName[k], gt_flow);
                gtsQueue.push_back(gt_flow);
            }

            if(imgsQueue.size() > 0) {
                for(int k = 1; k < imgsQueue.size(); k++) {
                    cv::Mat cur_img, forw_img;
                    cv::Mat flow_estim, flow_gt;
                    cur_img = imgsQueue[k - 1];
                    forw_img = imgsQueue[k];
                    flow_gt = gtsQueue[k - 1];
                    
                    cv::calcOpticalFlowFarneback(
                        cur_img, forw_img, flow_estim, 
                        0.5, 5, 15, 10, 3, 1.2, 
                        cv::OPTFLOW_FARNEBACK_GAUSSIAN);
                    
                    // cv::Ptr<cv::DenseOpticalFlow> deepFlow = cv::optflow::createOptFlow_DeepFlow();
                    // deepFlow -> calc(cur_img, forw_img, flow_estim);
                    
                    // show flow
                    cv::Mat color_estim, color_gt;
                    motionToColor(flow_estim, color_estim);
                    motionToColor(flow_gt, color_gt);
                    cv::imshow("estimate flow show", color_estim);
                    cv::imshow("gt flow show", color_gt);
                    cv::waitKey(10);
                }
            }
        }

        
    }
}