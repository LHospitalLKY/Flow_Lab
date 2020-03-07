//
// Author: LHO LHospitalLKY@github.com 
//

#include "../read_write.h"
#include "../evaluation.h"
#include "../show_flow.h"

int main(int argc, char *argv[]) {

    cv::Mat estim_flow, gt_flow, estim_color, gt_color;
    read(
        "/home/lho/SLAM/FlowNet/Flow_lab/CppLab/DataSet/test_result/ambush_2_0001.flo", 
        estim_flow
    );
    read(
        "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/MPI-Sintel-complete/training/flow/ambush_2/frame_0001.flo", 
        gt_flow
    );

    motionToColor(estim_flow, estim_color);
    motionToColor(gt_flow, gt_color);

    cv::Mat EPE_Mat(estim_flow.rows, estim_flow.cols, CV_32FC1);
    double epe_err = endPointError(estim_flow, gt_flow, EPE_Mat);

    cv::imshow("estimate", estim_color);
    cv::imshow("gt", gt_color);

    cv::imshow("epe error", EPE_Mat);
    cv::waitKey(0);
    std::cout << epe_err << std::endl;

}