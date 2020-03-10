//
// Author: LHO LHospitalLKY@github.com 
//

#include "line_tracking.h"

LineOpticalFlow::LineOpticalFlow() {

    deepflow_ = cv::optflow::createOptFlow_DeepFlow();
    lsd_ = cv::createLineSegmentDetector(LSD_REFINE_STD);

}

void LineOpticalFlow::calc(
    cv::Mat &prev, 
    cv::Mat &cur, 
    std::vector<cv::Vec4f> &prev_lines, 
    std::vector<cv::Vec4f> &cur_lines,
    std::vector<int> &status 
) {

    auto clock_start = std::chrono::system_clock::now();

    // 验证数据类型
    if(prev.type() != CV_8UC1 || cur.type() != CV_8UC1) {
        std::cerr << "Image Type is not CV_8UC1, please check your input!" << std::endl;
        return;
    }
    
    // 提取特征
    lsd_ -> detect(prev, prev_lines);

    auto clock_lines = std::chrono::system_clock::now();

    // 计算光流
    cv::Mat flow;
    deepflow_ -> calc(prev, cur, flow);

    auto clock_flow = std::chrono::system_clock::now();

    // 根据光流结果跟踪直线
    // 循环不是瓶颈，不到1ms的运算时间，瓶颈在flow中
    int num_lines = prev_lines.size();
    int x_max = flow.size().width;
    int y_max = flow.size().height;

    for(int i = 0; i < num_lines; i++) {
        double xs = (prev_lines)[i][0];
        double ys = (prev_lines)[i][1];
        double xe = (prev_lines)[i][2];
        double ye = (prev_lines)[i][3];
        
        bool in_bound_p = inBoundBox(xs, ys, x_max, y_max) && inBoundBox(xe, ye, x_max, y_max);
        if(in_bound_p == false) {
            continue;
        }

        // 找到对应的uv
        float *uv_s = flow.ptr<float>((int)ys, (int)xs);
        double xs_esti = xs + uv_s[0];
        double ys_esti = ys + uv_s[1];

        float *uv_e = flow.ptr<float>((int)ye, (int)xe);
        double xe_esti = xe + uv_e[0];
        double ye_esti = ye + uv_e[1];

        bool in_bound_c = inBoundBox(xs_esti, ys_esti, x_max, y_max) && inBoundBox(xe_esti, ye_esti, x_max, y_max);
        if(in_bound_c == false) {
            continue;
        }
        cv::Vec4f endPoints;
        endPoints << xs_esti, ys_esti, xe_esti, ye_esti;
        cur_lines.push_back(endPoints);
        status.push_back(1);
    }

    auto clock_end = std::chrono::system_clock::now();

    auto lines_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(clock_lines - clock_start);
    auto flow_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(clock_flow - clock_lines);
    auto tracking_time_used = std::chrono::duration_cast<std::chrono::microseconds>(clock_end - clock_flow);
    auto all_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start);

    std::cout << "All time uesd: " << all_time_used.count() << "ms" << std::endl;
    std::cout << "LineSegment time uesd: " << lines_time_used.count() << "ms" << std::endl;
    std::cout << "Flow calc time uesd: " << flow_time_used.count() << "ms" << std::endl;
    std::cout << "Tracking time uesd: " << tracking_time_used.count() << "mus" << std::endl;

    all_time_duration_ = all_time_used.count();
    flowCalc_time_duration_ = flow_time_used.count();
    lineSeg_time_duration_ = tracking_time_used.count();

}

bool LineOpticalFlow::inBoundBox(double x, double y, double x_max, double y_max) {
    if(x < 0 || y < 0) {
        return false;
    }
    if(x > x_max || y > y_max) {
        return false;
    }
    return true;
}

cv::Mat LineOpticalFlow::drawTracking(cv::Mat &cur, std::vector<cv::Vec4f> &cur_lines) {
    cv::Mat drawTracking(cur);
    lsd_ -> drawSegments(drawTracking, cur_lines);
    return drawTracking;
}

void LineOpticalFlow::returnDuration(Duration &durations) {
    durations.all_duration = all_time_duration_;
    durations.flow_duration = flowCalc_time_duration_;
    durations.line_duration = lineSeg_time_duration_;
}