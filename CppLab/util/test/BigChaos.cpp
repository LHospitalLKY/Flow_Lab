//
// Author: LHO LHospitalLKY@github.com 
//

#include "../Gradient.h"

#include <chrono>
#include <cmath>

#include <eigen3/Eigen/Eigen>

#define _USE_MATH_DEFINES

void LBD(cv::Mat &img, GradientMap gm);
void LBD2(cv::Mat &img, GradientMap gm);

int main(int argc, char *argv[]) {

    GradientMap gm;
    std::shared_ptr<Gradient> gradient = std::make_shared<Gradient>();

    cv::Mat frame_0 = cv::imread("/home/lho/SLAM/FlowNet/Flow_lab/Line Segment Detector/data/LK1.png", CV_8UC1);

    auto start_time = std::chrono::system_clock::now();
    gradient -> compute(frame_0);
    gm = gradient -> returnGradient();
    auto end_time = std::chrono::system_clock::now();

    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "My time uesd: " << time_used.count() << "ms" << std::endl;

    cv::imshow("gradient_level", gm.magnitudeLevel);
    cv::waitKey(0);

    LBD2(frame_0, gm);

    return 1;

}

void LBD(cv::Mat &img, GradientMap gm) {
    // 线分割
    std::vector<cv::line_descriptor::KeyLine> keylines;
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
    lsd -> detect(img, keylines, 5, 1, cv::Mat());
    std::cout << "Num of lsd detected: " << keylines.size() << std::endl;
    // 画图
    cv::Mat showLines;
    cv::line_descriptor::drawKeylines(img, keylines, showLines, cv::Scalar(0));
    cv::imshow("lines", showLines);
    
    // 找到一条长度比较长的线段
    int max_length = 0;
    cv::line_descriptor::KeyLine maxLengthKeyLine;
    for(auto &keyline : keylines) {
        if(keyline.lineLength > max_length) {
            max_length = keyline.lineLength;
            maxLengthKeyLine = keyline;
        }
    }
    std::cout << "Max Length: " << max_length << std::endl;

    // 计算出d_l与d_orth
    Eigen::Vector2d d_l, d_orth;
    double d_l_x, d_l_y;
    d_l_x = maxLengthKeyLine.getEndPoint().x - maxLengthKeyLine.getStartPoint().x;
    d_l_y = maxLengthKeyLine.getEndPoint().y - maxLengthKeyLine.getStartPoint().y;
    d_l << d_l_x, d_l_y;
    d_l.normalize();
    Eigen::Matrix2d orthMatrix;
    orthMatrix << 0, 1, -1, 0;
    d_orth = orthMatrix * d_l;

    std::cout << "d_l: " << d_l << std::endl;
    std::cout << "d_orth: " << d_orth << std::endl;
    std::cout << d_l.transpose() * d_orth << std::endl;

    Eigen::Vector2d center;
    Eigen::Matrix2d rotation;
    center << maxLengthKeyLine.pt.x, maxLengthKeyLine.pt.y;
    rotation << d_l[0], d_l[1], -d_l[1], d_l[0];

    Eigen::Vector2d start;
    start << maxLengthKeyLine.getStartPoint().x, maxLengthKeyLine.getStartPoint().y;

    Eigen::Vector2d end;
    end << maxLengthKeyLine.getEndPoint().x, maxLengthKeyLine.getEndPoint().y;

    std::cout << "Rotation matrix:\n" << rotation << std::endl;
    std::cout << "Center: " << center.transpose() << std::endl;
    std::cout << "Start before transform: " << start << std::endl;
    std::cout << "End before transform: " << end << std::endl;
    std::cout << "Start after rotation: " << rotation.transpose() * (start - center) << std::endl;

    Eigen::Vector2d bx_1, bx_2, bx_3, bx_4;
    bx_1 = start + 5 * -d_orth;
    bx_2 = end + 5 * -d_orth;
    bx_3 = start + 5 * d_orth;
    bx_4 = end + 5 * d_orth;

    std::cout << bx_1 << std::endl;
    std::cout << bx_2 << std::endl;
    std::cout << bx_3 << std::endl;
    std::cout << bx_4 << std::endl;

    std::cout << (bx_2 - bx_1).normalized().transpose() << std::endl;
    std::cout << (bx_4 - bx_3).normalized().transpose() << std::endl;

    // 找到全部的lsr
    int m = 5;
    int w = 3;
    std::vector<cv::Point2f> line_pixels;
    for(int i = 0; i < (int)(maxLengthKeyLine.lineLength + 0.5); i++) {
        Eigen::Vector2d pixel;
        pixel = start + i * d_l;
        line_pixels.push_back(cv::Point2f(pixel[0], pixel[1]));
    }
    std::cout << "line_pixel size: " << line_pixels.size() << std::endl;

    std::vector<cv::Point2f> lsr;
    for(int i = 0; i < 7; i++) {
        for(auto &line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            // 负d_orth方向
            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;
            // 正d_orth方向
            Eigen::Vector2d dx_2;
            dx_2 = origin + i * d_orth;

            lsr.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            lsr.push_back(cv::Point2f(dx_2[0], dx_2[1]));
        }
    }

    cv::Mat draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : lsr) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    cv::imshow("draw lsr", draw);

    auto start_band = std::chrono::system_clock::now();
    // band
    // std::vector<std::vector<cv::Point2f>> lsr;
    // band 1
    std::vector<cv::Point2f> band_1;
    std::vector<cv::Point2f> gradient_1;
    cv::Point2f center_1;
    double x_center_1, y_center_1;
    int count_1 = 0;
    for(int i = 5; i < 8; i++) {
        for(auto &line_pixel : line_pixels) {
            count_1++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(line_pixel.y, line_pixel.x), 
                           gm.dy.at<float>(line_pixel.y, line_pixel.x);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;

            band_1.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            gradient_1.push_back(cv::Point2f(grad_band[0], grad_band[1]));

            x_center_1 += dx_1[0];
            y_center_1 += dx_1[1];
        }
    }
    x_center_1 = x_center_1/(float)count_1;
    y_center_1 = y_center_1/(float)count_1;
    center_1.x = x_center_1;
    center_1.y = y_center_1;
    draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : band_1) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    std::cout << "band 1 center: " << center_1 << std::endl;
    cv::imshow("draw lsr 1", draw);

    // band 2
    std::vector<cv::Point2f> band_2;
    std::vector<cv::Point2f> gradient_2;
    cv::Point2f center_2;
    double x_center_2, y_center_2;
    int count_2 = 0;
    for(int i = 2; i < 5; i++) {
        for(auto &line_pixel : line_pixels) {
            count_2++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(line_pixel.y, line_pixel.x), 
                           gm.dy.at<float>(line_pixel.y, line_pixel.x);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;

            band_2.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            gradient_2.push_back(cv::Point2f(grad_band[0], grad_band[1]));
            x_center_2 += dx_1[0];
            y_center_2 += dx_1[1];
        }
    }
    x_center_2 = x_center_2/(float)count_2;
    y_center_2 = y_center_2/(float)count_2;
    center_2.x = x_center_2;
    center_2.y = y_center_2;
    draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : band_2) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    std::cout << "band 2 center: " << center_2 << std::endl;
    cv::imshow("draw lsr 2", draw);

    // band 3
    std::vector<cv::Point2f> band_3;
    std::vector<cv::Point2f> gradient_3;
    cv::Point2f center_3;
    double x_center_3, y_center_3;
    int count_3 = 0;
    for(int i = -1; i < 2; i++) {
        for(auto &line_pixel : line_pixels) {
            count_3++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(line_pixel.y, line_pixel.x), 
                           gm.dy.at<float>(line_pixel.y, line_pixel.x);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;

            band_3.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            gradient_3.push_back(cv::Point2f(grad_band[0], grad_band[1]));
            x_center_3 += dx_1[0];
            y_center_3 += dx_1[1];
        }
    }
    x_center_3 = x_center_3/(float)count_3;
    y_center_3 = y_center_3/(float)count_3;
    center_3.x = x_center_3;
    center_3.y = y_center_3;
    draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : band_3) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    std::cout << "band 3 center: " << center_3 << std::endl;
    cv::imshow("draw lsr 3", draw);

    // band 4
    std::vector<cv::Point2f> band_4;
    std::vector<cv::Point2f> gradient_4;
    cv::Point2f center_4;
    double x_center_4, y_center_4;
    int count_4 = 0;
    for(int i = -4; i < -1; i++) {
        for(auto &line_pixel : line_pixels) {
            count_4++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(line_pixel.y, line_pixel.x), 
                           gm.dy.at<float>(line_pixel.y, line_pixel.x);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;

            band_4.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            gradient_4.push_back(cv::Point2f(grad_band[0], grad_band[1]));
            x_center_4 += dx_1[0];
            y_center_4 += dx_1[1];
        }
    }
    x_center_4 = x_center_4/(float)count_4;
    y_center_4 = y_center_4/(float)count_4;
    center_4.x = x_center_4;
    center_4.y = y_center_4;
    draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : band_4) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    std::cout << "band 4 center: " << center_4 << std::endl;
    cv::imshow("draw lsr 4", draw);

    // band 5
    std::vector<cv::Point2f> band_5;
    std::vector<cv::Point2f> gradient_5;
    cv::Point2f center_5;
    double x_center_5, y_center_5;
    int count_5 = 0;
    for(int i = -7; i < -4; i++) {
        for(auto &line_pixel : line_pixels) {
            count_5++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d dx_1;
            dx_1 = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(line_pixel.y, line_pixel.x), 
                           gm.dy.at<float>(line_pixel.y, line_pixel.x);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;

            band_5.push_back(cv::Point2f(dx_1[0], dx_1[1]));
            gradient_5.push_back(cv::Point2f(grad_band[0], grad_band[1]));
            x_center_5 += dx_1[0];
            y_center_5 += dx_1[1];
        }
    }
    x_center_5 = x_center_5/(float)count_5;
    y_center_5 = y_center_5/(float)count_5;
    center_5.x = x_center_5;
    center_5.y = y_center_5;
    draw = cv::Mat::zeros(img.size(), CV_8UC1);
    for(auto &point : band_5) {
        draw.at<uchar>(point.y, point.x) = 255;
    }
    std::cout << "band 5 center: " << center_5 << std::endl;
    cv::imshow("draw lsr 5", draw);

    auto end_band = std::chrono::system_clock::now();
    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_band - start_band);
    std::cout << "Band time uesd: " << time_used.count() << "ms" << std::endl;

    cv::waitKey(0);
    
}

struct Band {
    cv::Mat LSR_gx;
    cv::Mat LSR_gy;
    cv::Point2f center;
    std::vector<float> global_weight;
    std::vector<float> local_weight;
};

void LBD2(cv::Mat &img, GradientMap gm) {

    // 找到线段的位置
    // 线分割
    std::vector<cv::line_descriptor::KeyLine> keylines;
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
    lsd -> detect(img, keylines, 5, 1, cv::Mat());
    std::cout << "Num of lsd detected: " << keylines.size() << std::endl;
    // 画图
    cv::Mat showLines;
    cv::line_descriptor::drawKeylines(img, keylines, showLines, cv::Scalar(0));
    cv::imshow("lines", showLines);
    
    // 找到一条长度比较长的线段
    int max_length = 0;
    cv::line_descriptor::KeyLine maxLengthKeyLine;
    for(auto &keyline : keylines) {
        if(keyline.lineLength > max_length) {
            max_length = keyline.lineLength;
            maxLengthKeyLine = keyline;
        }
    }
    std::cout << "Max Length: " << max_length << std::endl;
    // 计算出d_l与d_orth
    Eigen::Vector2d d_l, d_orth;
    double d_l_x, d_l_y;
    d_l_x = maxLengthKeyLine.getEndPoint().x - maxLengthKeyLine.getStartPoint().x;
    d_l_y = maxLengthKeyLine.getEndPoint().y - maxLengthKeyLine.getStartPoint().y;
    d_l << d_l_x, d_l_y;
    d_l.normalize();
    Eigen::Matrix2d orthMatrix;
    orthMatrix << 0, 1, -1, 0;
    d_orth = orthMatrix * d_l;
    // 找到线段区域
    Eigen::Vector2d start;
    start << maxLengthKeyLine.getStartPoint().x, maxLengthKeyLine.getStartPoint().y;

    Eigen::Vector2d end;
    end << maxLengthKeyLine.getEndPoint().x, maxLengthKeyLine.getEndPoint().y;

    std::vector<cv::Point2f> line_pixels;
    for(int i = 0; i < (int)(maxLengthKeyLine.lineLength + 0.5); i++) {
        Eigen::Vector2d pixel;
        pixel = start + i * d_l;
        line_pixels.push_back(cv::Point2f(pixel[0], pixel[1]));
    }
    std::cout << "line_pixel size: " << line_pixels.size() << std::endl;

    // TODO: 这是每个band的计算
    // 每个band采用一个结构体来存储
    Band band1, band2, band3, band4;
    int m = 5;
    int w = 3;
    double theta_g = 0.5 * (m * w - 1);
    double theta_l = w;
    // band 1
    double x_center_1, y_center_1;
    int count_1 = 0;
    band1.LSR_gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band1.LSR_gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    cv::Point2f global_center = maxLengthKeyLine.pt;
    int count_row, count_col;
    count_row = 0;
    count_col = 0;
    for(int i = 5; i < 8; i++) {
        count_col = 0;

        // 全局weight
        double global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band1.global_weight.push_back(global_weight_tmp);
        // 局部weight
        double local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band1.local_weight.push_back(local_weight_tmp);

        for(auto &line_pixel : line_pixels) {
            count_1++;
            Eigen::Vector2d origin;
            origin << line_pixel.x, line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth;

            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), 
                           gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth, grad_origin.transpose() * d_l;
            
            // 局部坐标梯度
            band1.LSR_gx.at<float>(count_col, count_row) = grad_band[0] + 255;
            band1.LSR_gy.at<float>(count_col, count_row) = grad_band[1] + 255;

            x_center_1 += point[0];
            y_center_1 += point[1];

            count_col++;

            std::cout << "===================" << std::endl;
            std::cout << "Grandient: " << grad_band << std::endl;
        }
        count_row++;
    }
    band1.center.x = x_center_1/count_1;
    band1.center.y = y_center_1/count_1;

    std::cout << ">>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "Global weight 0: " << band1.global_weight[0] << std::endl;
    std::cout << "Global weight 1: " << band1.global_weight[1] << std::endl;
    std::cout << "Global weight 2: " << band1.global_weight[2] << std::endl;
    std::cout << "Local weight 0: " << band1.local_weight[0] << std::endl;
    std::cout << "Local weight 1: " << band1.local_weight[1] << std::endl;
    std::cout << "Local weight 2: " << band1.local_weight[2] << std::endl;
    
    cv::imshow("band1 gx", band1.LSR_gx);
    cv::imshow("band1 gy", band1.LSR_gy);
    cv::imshow("band1 global weight", band1.global_weight);

    cv::waitKey(0);

}