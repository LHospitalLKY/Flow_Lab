//
// Author: LHO LHospitalLKY@github.com 
//

#include "../show_flow.cpp"
#include "../read_write.h"

#include <iostream>

int main(int argc, char *argv[]) {

    // read
    std::string floName = "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/MPI-Sintel-complete/training/flow/ambush_2/frame_0001.flo";
    cv::Mat flow_read;
    read(floName, flow_read);

    // 可视化
    cv::Mat color;
    motionToColor(flow_read, color);
    cv::imshow("flow_read", color);
    cv::waitKey(0);

    // write
    std::string writeName = "/home/lho/SLAM/FlowNet/Flow_lab/CppLab/DataSet/test_result/MPI_Clean_Estimate/ambush_2_0001.flo";
    write(writeName, flow_read);
    
    return 1;
}