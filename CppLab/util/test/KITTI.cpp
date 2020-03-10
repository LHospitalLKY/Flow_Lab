//
// Author: LHO LHospitalLKY@github.com 
//

#include "../read_write.h"
#include "../show_flow.h"

#include <cmath>
#include <eigen3/Eigen/Dense>
#include <opencv2/optflow.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/line_descriptor.hpp>

int main(int argc, char *argv[]) {

    // TUM
    // cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/TUM/f-xyz1/1/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png");
    // cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/TUM/f-xyz1/1/rgbd_dataset_freiburg1_xyz/rgb/1305031102.211214.png");

    // EUROC
    // cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638523377829376.png", CV_8UC1);
    // cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638523527829504.png", CV_8UC1);

    // KITTI
    cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000000_10.png");
    cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000000_11.png");

    cv::imshow("f1", frame_0);
    cv::imshow("f2", frame_1);

    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);
    std::cout << frame_0_gray.type() << std::endl;
    std::cout << CV_8UC2 << std::endl;
    // frame_0_gray = frame_0.clone();
    // frame_1_gray = frame_1.clone();

    cv::Mat flow;
    cv::Ptr<cv::DenseOpticalFlow> deepflow = cv::optflow::createOptFlow_DeepFlow();
    deepflow -> calc(frame_0_gray, frame_1_gray, flow);

    cv::Mat color;
    motionToColor(flow, color);

    cv::imshow("flow", color);
    cv::waitKey(0);

    // 提取直线
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(LSD_REFINE_STD);
    std::vector<cv::Vec4f> lines;
    lsd -> detect(frame_0_gray, lines);
    cv::Mat drawLines(frame_0);
    lsd -> drawSegments(drawLines, lines);

    cv::imshow("lines", drawLines);
    cv::waitKey(0);

    // 根据光流计算出位移后的位置
    int num_lines = lines.size();
    std::vector<cv::Vec4f> lines_esti;
    int x_max = flow.size().width;
    int y_max = flow.size().height;
    for(int i = 0; i < num_lines; i++) {
        double x_s = lines[i][0];
        double y_s = lines[i][1];
        double x_e = lines[i][2];
        double y_e = lines[i][3];
        if(x_s < 0 || y_s < 0 || x_e < 0 || y_e < 0) {
            continue;
        }
        if(x_s > x_max || y_s > y_max || x_e > x_max || y_e > y_max) {
            continue;
        }
        // 找到u, v
        float *uv_s = flow.ptr<float>((int)y_s, (int)x_s);
        std::cout << uv_s[0] << " " << uv_s[1] << std::endl;
        double x_s_es = x_s + uv_s[0];
        double y_s_es = y_s + uv_s[1];
        float *uv_e = flow.ptr<float>((int)y_e, (int)x_e);
        std::cout << uv_e[0] << " " << uv_e[1] << std::endl;
        double x_e_es = x_e + uv_e[0];
        double y_e_es = y_e + uv_e[1];

        cv::Vec4f endPoint;
        endPoint << x_s_es, y_s_es, x_e_es, y_e_es;
        lines_esti.push_back(endPoint);

        std::cout << x_s << " " <<
                    y_s <<  " " <<
                    x_e <<  " " <<
                    y_e << std::endl;
        std::cout << x_s_es << " " <<
                    y_s_es <<  " " <<
                    x_e_es <<  " " <<
                    y_e_es << std::endl;
        std::cout << endPoint << std::endl;
        std::cout << *(lines_esti.end() - 1) << std::endl;
        std::cout << "----------------------------" << std::endl;

    }

    std::cout << lines_esti.size() << std::endl;

    cv::Mat drawLines_2(frame_1);
    lsd -> drawSegments(drawLines_2, lines_esti);
    cv::imshow("line estimate", drawLines_2);
    cv::waitKey(0);


    return 1;

}