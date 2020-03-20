//
// Author: LHO LHospitalLKY@github.com 
//

/**
 * kitti 2015 flow datasets lines tracking experiment
 **/

#include "../util/line_tracking.h"
#include "../util/read_write.h"
#include "../util/show_flow.h"

#include <iostream>

/**
 * @brief: 读取数据集的数据，并进行一些预处理工作
 * 读取的数据包括：
 * 1. image pair，用于估计光流
 * 2. semantic mask，用于滤掉车辆
 * 要进行的预处理：
 * 1. 为frame_0加上mask
 * 输出的值：
 * frame_0, frame_1
 * mask
 **/
class DataSetReader {
public:
    DataSetReader(){
        max_num_ = 200;
        batch_size_ = 4;
    }
    
    // 设置数据路径
    void setRootPath(std::string rgb_path, std::string mask_path) {
        rgb_path_ = rgb_path;
        mask_path_ = mask_path;
    }

    /**
     * @brief: 读取数据
     * @param: 
     *  1. pointer, 整型，队列中第一个图像的id
     *  2. rgb_vector, 返回batch_size个rgb图像pair
     *  3. mask_vector, 返回batch_size个mask
     *  4. 
     * @return: 
     * @ret: 
     **/
    void readImages(int pointer, std::vector<cv::Mat>) {

    }

public:
    std::string rgb_path_;
    std::string mask_path_;

    int max_num_;
    int batch_size_;
};

int main(int argc, char *argv[]) {

    std::cout << "Hello World!" << std::endl;
    return 1;

}