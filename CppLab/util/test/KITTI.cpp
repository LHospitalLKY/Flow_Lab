//
// Author: LHO LHospitalLKY@github.com 
//

#include "../read_write.h"
#include "../show_flow.h"
#include "../line_tracking.h"

#include <cmath>
#include <memory>
#include <eigen3/Eigen/Dense>
#include <opencv2/optflow.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/line_descriptor.hpp>
#include <png++/png.hpp>

/*
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
*/

int main(int argc, char *argv[]) {
    // KITTI
    cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000005_10.png");
    cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/image_2/000005_11.png");
    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);
    
    std::string flow_gt_path = "/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_scene_flow/training/flow_noc/000000_10.png";

    int width = frame_0.size().width;
    int height = frame_1.size().height;
    png::image<png::rgb_pixel_16> image(flow_gt_path);
    cv::Mat flow_gt = cv::Mat::zeros(cv::Size(width, height), CV_32FC2);
    cv::Mat flow_valid = cv::Mat::zeros(cv::Size(width, height), CV_16UC1);
    int count = 0;
    for(int v = 0; v < height; v++) {
        for(int u = 0; u < width; u++) {
            float *data = flow_gt.ptr<float>(v, u);
            png::rgb_pixel_16 val = image.get_pixel(u, v);
            if(val.blue > 0) {
                count++;
                data[0] = ((float)val.red - 32768.0f)/64.0f;
                data[1] = ((float)val.green - 32768.0f)/64.0;
                flow_valid.at<int>(v, u) = val.blue;
                // data[2] = val.blue;
                // std::cout << "Valid: " << data[2] << std::endl;
                std::cout << "u: " << data[0] << std::endl;
                std::cout << "v: " << data[1] << std::endl;
                std::cout << "------------" << std::endl;
            } else {
                data[0] = 0;
                data[1] = 0;
            }
            
            // data[2] = val.blue;
        }
    }

    std::cout << "valid count: " << count << std::endl;
    std::cout << "flow map size: " << flow_gt.size() << std::endl;

    cv::Mat color;
    motionToColor(flow_gt, color);

    std::vector<cv::Mat> flow_split;
    cv::split(flow_gt, flow_split);

    cv::imshow("flow", color);
    // cv::waitKey(0);

    // 对gt进行卷积操作
    // TODO: 平滑不靠谱
    /*
    cv::Mat flow_filter;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) << 1, 1, 1, 
                                                1, 1, 1,
                                                1, 1, 1);
    cv::filter2D(flow_gt, flow_filter, -1, kernel);
    cv::Mat color_filter;
    motionToColor(flow_filter, color_filter);
    cv::imshow("flow_filter", color_filter);
    cv::waitKey(0);
    */

    // 读取instance_mat
    png::image<png::gray_pixel> instance_mat("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_semantics/training/instance/000001_10.png");
    png::image<png::gray_pixel> semantic_mat("/media/lho/064A027D79FA99C7/slam数据集/光流数据集/KITTI2015/data_semantics/training/semantic/000001_10.png");

    cv::Mat instance(frame_0.size(), CV_8UC1);
    cv::Mat semantic(frame_1.size(), CV_8UC1);
    uint min_instance = 999;
    uint max_instance = 0;
    uint min_semantic = 999;
    uint max_semantic = 0;
    for(int i = 0; i < frame_0.cols; i++) {
        for(int j = 0; j < frame_0.rows; j++) {
            png::gray_pixel pixel_ins, pixel_sem;
            pixel_ins = instance_mat.get_pixel(i, j);
            pixel_sem = semantic_mat.get_pixel(i, j);
            instance.at<uchar>(j, i) = (uchar)pixel_ins % 256;
            semantic.at<uchar>(j, i) = (uchar)pixel_sem;
            // std::cout << "instance: " << (uchar)pixel_ins % 256 << std::endl;
            // std::cout << "semantic: " << (uchar)pixel_sem << std::endl;
            if(instance.at<uchar>(j, i) > max_instance)
                max_instance = instance.at<uchar>(j, i);
            if(instance.at<uchar>(j, i) < min_instance)
                min_instance = instance.at<uchar>(j, i);
            if(semantic.at<uchar>(j, i) > max_semantic)
                max_semantic = semantic.at<uchar>(j, i);
            if(semantic.at<uchar>(j, i) < min_semantic)
                min_semantic = semantic.at<uchar>(j, i);
        }
    }

    std::cout << "instance label:" << min_instance << "~" << max_instance << std::endl; 
    std::cout << "semantic label:" << min_semantic << "~" << max_semantic << std::endl;

    // cv::imshow("instance", instance);
    // cv::imshow("semantic", semantic);
    // cv::waitKey(0);

    // 取出label = 4的位置
    std::vector<cv::Point2f> label4_index;
    int count_label4 = 0;
    for(int i = 0; i < instance.cols; i++) {
        for(int j = 0; j < instance.rows; j++) {
            if(instance.at<uchar>(j, i) == 26) {
                label4_index.push_back(cv::Point2f(j, i));
                count_label4++;
            }
        }
    }

    std::cout << "num of label 4: " << count_label4 << std::endl;

    // 画出label 4的内容
    cv::Mat isolate_4 = cv::Mat::zeros(frame_0.size(), CV_8UC1);
    for(int i = 0; i < label4_index.size(); i++) {
        double x = label4_index[i].x;
        double y = label4_index[i].y;

        isolate_4.at<uchar>(x, y) = 255;
        
    }

    // cv::imshow("label 26", isolate_4);
    // cv::waitKey(0);

    // 找到label 26中存在光流gt的点
    std::vector<cv::Point2f> car_valid;
    std::vector<cv::Point2f> car_valid_esti;
    int count_car_valid = 0;
    for(int i = 0; i < label4_index.size(); i++) {
        float x = label4_index[i].x;
        float y = label4_index[i].y;
        png::rgb_pixel_16 pixel = image.get_pixel(y, x);
        if(pixel.blue > 0) {
            car_valid.push_back(cv::Point2f(x, y));
            float *speed = flow_gt.ptr<float>(x, y);
            x = x + speed[1];
            y = y + speed[0];
            car_valid_esti.push_back(cv::Point2f(x, y));
            count_car_valid++;
        }
    }
    std::cout << "car valid: " << car_valid.size() << std::endl;
    std::cout << "car valid est: " << car_valid_esti.size() << std::endl;

    cv::Mat car_valid_mat = cv::Mat::zeros(frame_0.size(), CV_8UC1);
    cv::Mat car_valid_esti_mat = cv::Mat::zeros(frame_0.size(), CV_8UC1);
    for(int i = 0; i < car_valid.size(); i++) {
        car_valid_mat.at<uchar>(car_valid[i].x, car_valid[i].y) = 255;
        if(car_valid_esti[i].x < 0 || car_valid_esti[i].x > frame_0.rows) {
            continue;
        }
        if(car_valid_esti[i].y < 0 || car_valid_esti[i].y > frame_0.cols) {
            continue;
        }
        car_valid_esti_mat.at<uchar>(car_valid_esti[i].x, car_valid_esti[i].y) = 255;
    }
    // 根据光流把这些点推到下一幅图中


    // cv::imshow("car flow valid", car_valid_mat);
    // cv::imshow("car estim", car_valid_esti_mat);

    // 将空位填补
    int min_dist = 5;
    cv::Mat mask(frame_0.size(), CV_8UC1, cv::Scalar(255));
    cv::Mat mask_esti(frame_0.size(), CV_8UC1, cv::Scalar(255));
    for(auto &it : car_valid) {
        if(car_valid_mat.at<uchar>(it.x, it.y) == 255) {
            cv::circle(mask, cv::Point2f(it.y, it.x), min_dist, 0, -1);
        }
    }
    for(auto &it : car_valid_esti) {
        if(car_valid_esti_mat.at<uchar>(it.x, it.y) == 255) {
            cv::circle(mask_esti, cv::Point2f(it.y, it.x), min_dist, 0, -1);
        }
    }

    // cv::imshow("mask", mask);
    // cv::imshow("mask_esti", mask_esti);

    // 为原图加mask
    /*
    cv::Mat frame_0_mask = frame_0.setTo(0, mask);
    cv::Mat frame_1_mask = frame_1.setTo(0, mask_esti);
    cv::Mat frame_0_mask_gray, frame_1_mask_gray;
    cv::cvtColor(frame_0_mask, frame_0_mask_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1_mask, frame_1_mask_gray, CV_BGR2GRAY);

    cv::imshow("mask frame_0", frame_0_mask);
    cv::imshow("mask frame_1", frame_1_mask);
    */

    // 在mask上进行光流
    /*
    cv::Mat flow_mask;
    cv::Mat flow_mask_color;
    cv::Ptr<cv::DenseOpticalFlow> deepflow = cv::optflow::createOptFlow_DeepFlow();
    deepflow -> calc(frame_0_mask_gray, frame_1_mask_gray, flow_mask);
    motionToColor(flow_mask, flow_mask_color);

    cv::Mat flow_esti;
    cv::Mat flow_esti_color;
    deepflow -> calc(frame_0_gray, frame_1_gray, flow_esti);
    motionToColor(flow_esti, flow_esti_color);

    cv::imshow("mask flow", flow_mask_color);
    cv::imshow("origin flow", flow_esti_color);

    cv::waitKey(0);
    */

    std::shared_ptr<LineOpticalFlow> lof = std::make_shared<LineOpticalFlow>();
    std::vector<cv::Vec4f> prev_lines;
    std::vector<cv::Vec4f> cur_lines;
    std::vector<uchar> status;

    lof -> calc(frame_0_gray, frame_1_gray, prev_lines, cur_lines, status);
    // cv::imshow("frame_0", frame_0);
    cv::Mat draw_0, draw_1;
    draw_0 = lof -> drawTracking(frame_0, prev_lines);
    draw_1 = lof -> drawTracking(frame_1, cur_lines);
    cv::imshow("origin lines", draw_0);
    cv::imshow("tracking lines", draw_1);
    
    cv::waitKey(0);

    return 1;
}