//
// Author: LHO LHospitalLKY@github.com 
//

#include "../line_tracking.h"

#include <memory>

int main(int argc, char *argv[]) {

    std::shared_ptr<LineOpticalFlow> lof = std::make_shared<LineOpticalFlow>();

    // 读取图片
    cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638518077829376.png");
    cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638518227829504.png");

    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);

    std::vector<cv::Vec4f> prev_lines;
    std::vector<cv::Vec4f> cur_lines;
    std::vector<int> status;

    lof -> calc(frame_0_gray, frame_1_gray, prev_lines, cur_lines, status);
    cv::Mat draw;
    draw = lof -> drawTracking(frame_1, cur_lines);

    cv::imshow("draw tracking", draw);
    cv::waitKey(0);

}