//
// Author: LHO LHospitalLKY@github.com 
//

#include "LBD.h"

LSR::LSR() {
    // 必要的参数
    m = 5;
    w = 3;
    theta_g = 0.5 * (m * w - 1);
    theta_l = w;
}

void LSR::computeBands(const cv::Mat &img, const cv::line_descriptor::KeyLine &keyline, const GradientMap &gm) {
    // 线段中点
    global_center_ = keyline.pt;
    // 线段长度
    length_ = keyline.lineLength;
    // 线段的起点和中点
    start_point_ = keyline.getStartPoint();
    end_point_ = keyline.getEndPoint();
    // 线段的方向
    d_l_ << end_point_.x - start_point_.x, 
           end_point_.y - start_point_.y;
    d_l_.normalize();
    Eigen::Matrix2d orthMatrix;
    orthMatrix << 0, 1, -1, 0;
    d_orth_ = orthMatrix * d_l_;

    // 计算local center
    computeLocalCenter();

    // 计算梯度
    // 找到直线
    std::vector<cv::Point2f> line_pixels;
    for(int i = 0; i < (int)(keyline.lineLength + 0.5); i++) {
        Eigen::Vector2d start_point_E, pixel;
        start_point_E << start_point_.x, start_point_.y;
        pixel = start_point_E + i * d_l_;
        line_pixels.push_back(cv::Point2f(pixel[0], pixel[1]));
    }
    computeGradient(line_pixels, gm);

    // 计算描述子
    LBD_ = computeDescriptor();
    LBD_ = LBD_/length_;

}

void LSR::computeBands(const cv::Mat &img, const cv::Vec4f &keyline, const GradientMap &gm) {
    // 线段中点
    global_center_ = 0.5 * cv::Point2f(keyline[0] + keyline[2], keyline[1] + keyline[3]);
    // 线段的起点和中点
    start_point_ = cv::Point2f(keyline[0], keyline[1]);
    end_point_ = cv::Point2f(keyline[2], keyline[3]);
    // 线段长度
    length_ = std::sqrt(std::pow(start_point_.x - end_point_.x, 2) + std::pow(start_point_.y - end_point_.y, 2));
    // 线段的方向
    d_l_ << end_point_.x - start_point_.x, 
           end_point_.y - start_point_.y;
    d_l_.normalize();
    Eigen::Matrix2d orthMatrix;
    orthMatrix << 0, 1, -1, 0;
    d_orth_ = orthMatrix * d_l_;

    // 计算local center
    computeLocalCenter();

    // 计算梯度
    // 找到直线
    std::vector<cv::Point2f> line_pixels;
    for(int i = 0; i < (int)(length_ + 0.5); i++) {
        Eigen::Vector2d start_point_E, pixel;
        start_point_E << start_point_.x, start_point_.y;
        pixel = start_point_E + i * d_l_;
        line_pixels.push_back(cv::Point2f(pixel[0], pixel[1]));
    }
    computeGradient(line_pixels, gm);

    // 计算描述子
    LBD_ = computeDescriptor();

}

void LSR::computeLocalCenter() {
    
    Eigen::Vector2d global_center_eigen;
    global_center_eigen << global_center_.x, global_center_.y;

    // 找到各个band的中点
    cv::Point2f local_center_1, local_center_2, local_center_3, local_center_4, local_center_5;
    Eigen::Vector2d local_center_tmp;

    // band1中点
    local_center_tmp = global_center_eigen + 6 * -d_orth_;
    band1_.center = cv::Point2f(local_center_tmp[0], local_center_tmp[1]);
    // band2中点
    local_center_tmp = global_center_eigen + 3 * -d_orth_;
    band2_.center = cv::Point2f(local_center_tmp[0], local_center_tmp[1]);
    // band3中点
    local_center_tmp = global_center_eigen + 0 * -d_orth_;
    band3_.center = cv::Point2f(local_center_tmp[0], local_center_tmp[1]);
    // band4中点
    local_center_tmp = global_center_eigen + (-3) * -d_orth_;
    band4_.center = cv::Point2f(local_center_tmp[0], local_center_tmp[1]);
    // band5中点
    local_center_tmp = global_center_eigen + (-6) * -d_orth_;
    band5_.center = cv::Point2f(local_center_tmp[0], local_center_tmp[1]);

#ifdef Debug
    std::cout << "Band 1 Center: " << band1_.center << std::endl;
    std::cout << "Band 2 Center: " << band2_.center << std::endl;
    std::cout << "Band 3 Center: " << band3_.center << std::endl;
    std::cout << "Band 4 Center: " << band4_.center << std::endl;
    std::cout << "Band 5 Center: " << band5_.center << std::endl;
    std::cout << "Global Center: " << global_center_ << std::endl;
    std::cout << "dl:  " << d_l_ << std::endl;
#endif

}

void LSR::computeGradient(std::vector<cv::Point2f> &line_pixels, const GradientMap &gm) {
    
    // band1
    int count_row, count_col;
    count_row = 0;
    count_col = 0;
    band1_.gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band1_.gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    for(int i = 5; i < 8; i++) {
        count_col = 0;
        // 全局weight
        float global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band1_.global_weight.push_back(global_weight_tmp);
        // 局部weight
        float local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band1_.local_weight.push_back(local_weight_tmp);

        // 计算梯度
        for(auto line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << (double)line_pixel.x, (double)line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth_;
            // TODO: 检查index的顺序是否正确
            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth_, grad_origin.transpose() * d_l_;

            // 局部梯度
            band1_.gx.at<float>(count_row, count_col) = grad_band[0];
            band1_.gy.at<float>(count_row, count_col) = grad_band[1];

            count_col++;
        }
        count_row++;
    }

    // band2
    count_row = 0;
    count_col = 0;
    band2_.gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band2_.gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    for(int i = 2; i < 5; i++) {
        count_col = 0;
        // 全局weight
        float global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band2_.global_weight.push_back(global_weight_tmp);
        // 局部weight
        float local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band2_.local_weight.push_back(local_weight_tmp);

        // 计算梯度
        for(auto line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << (double)line_pixel.x, (double)line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth_;
            // TODO: 检查index的顺序是否正确
            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth_, grad_origin.transpose() * d_l_;

            // 局部梯度
            band2_.gx.at<float>(count_row, count_col) = grad_band[0];
            band2_.gy.at<float>(count_row, count_col) = grad_band[1];

            count_col++;
        }
        count_row++;
    }

    // band3
    count_row = 0;
    count_col = 0;
    band3_.gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band3_.gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    for(int i = -1; i < 2; i++) {
        count_col = 0;
        // 全局weight
        float global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band3_.global_weight.push_back(global_weight_tmp);
        // 局部weight
        float local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band3_.local_weight.push_back(local_weight_tmp);

        // 计算梯度
        for(auto line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << (double)line_pixel.x, (double)line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth_;
            // TODO: 检查index的顺序是否正确
            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth_, grad_origin.transpose() * d_l_;

            // 局部梯度
            band3_.gx.at<float>(count_row, count_col) = grad_band[0];
            band3_.gy.at<float>(count_row, count_col) = grad_band[1];

            count_col++;
        }
        count_row++;
    }

    // band4
    count_row = 0;
    count_col = 0;
    band4_.gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band4_.gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    for(int i = -4; i < -1; i++) {
        count_col = 0;
        // 全局weight
        float global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band4_.global_weight.push_back(global_weight_tmp);
        // 局部weight
        float local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band4_.local_weight.push_back(local_weight_tmp);

        // 计算梯度
        for(auto line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << (double)line_pixel.x, (double)line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth_;
            // TODO: 检查index的顺序是否正确
            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth_, grad_origin.transpose() * d_l_;

            // 局部梯度
            band4_.gx.at<float>(count_row, count_col) = grad_band[0];
            band4_.gy.at<float>(count_row, count_col) = grad_band[1];

            count_col++;
        }
        count_row++;
    }

    // band5
    count_row = 0;
    count_col = 0;
    band5_.gx = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    band5_.gy = cv::Mat::zeros(3, line_pixels.size(), CV_32FC1);
    for(int i = -7; i < -4; i++) {
        count_col = 0;
        // 全局weight
        float global_weight_tmp;
        global_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_g)) * std::exp(-std::pow(std::abs(i), 2)/(2*theta_g*theta_g));
        band5_.global_weight.push_back(global_weight_tmp);
        // 局部weight
        float local_weight_tmp;
        local_weight_tmp = (1/(std::sqrt(2*M_PI)*theta_l)) * std::exp(-std::pow(std::abs(count_row - 1), 2)/(2*theta_g*theta_g));
        band5_.local_weight.push_back(local_weight_tmp);

        // 计算梯度
        for(auto line_pixel : line_pixels) {
            Eigen::Vector2d origin;
            origin << (double)line_pixel.x, (double)line_pixel.y;

            Eigen::Vector2d point;
            point = origin + i * -d_orth_;
            // TODO: 检查index的顺序是否正确
            Eigen::Vector2d grad_origin, grad_band;
            grad_origin << gm.dx.at<float>(point[1], point[0]), gm.dy.at<float>(point[1], point[0]);
            grad_band << grad_origin.transpose() * d_orth_, grad_origin.transpose() * d_l_;

            // 局部梯度
            band5_.gx.at<float>(count_row, count_col) = grad_band[0];
            band5_.gy.at<float>(count_row, count_col) = grad_band[1];

            count_col++;
        }
        count_row++;
    }

}

MatrixBD LSR::computeDescriptor() {
    std::vector<Band> bands;
    bands.push_back(band1_);
    bands.push_back(band2_);
    bands.push_back(band3_);
    bands.push_back(band4_);
    bands.push_back(band5_);

    // 计算每个band的v_b
    std::vector<Matrix43, Eigen::aligned_allocator<Matrix43>> v_bands;
    for(int i = 0; i < bands.size(); i++) {
        Matrix43 v_b = computeDescriptor_singleBand(bands[i]);
        v_bands.push_back(v_b);
    }

    // 填充
    MatrixBD LBD;
    for (int i = 0; i < bands.size(); i++) {
        Eigen::Matrix<double, 4, 1> mean = Eigen::Matrix<double, 4, 1>::Zero();
        Eigen::Matrix<double, 4, 1> deviation = Eigen::Matrix<double, 4, 1>::Zero();
        if(i == 0) {
            Eigen::Matrix<double, 4, 6> BDM;
            BDM.block(0, 0, 4, 3) = v_bands[i];
            BDM.block(0, 3, 4, 3) = v_bands[i + 1];
            // std::cout << "BDM: " << std::endl;
            // std::cout << BDM << std::endl;

            mean = BDM.rowwise().mean();
            for(int a = 0; a < 6; a++) {
                deviation = deviation + ((BDM.col(a) - mean).array() * (BDM.col(a) - mean).array()).matrix();
            }
            deviation = (deviation/5).array().sqrt().matrix();
            // std::cout << "std" << std::endl;
            // std::cout << deviation.size() << std::endl;
        }
        if(i == bands.size() - 1) {
            Eigen::Matrix<double, 4, 6> BDM;
            BDM.block(0, 0, 4, 3) = v_bands[i - 1];
            BDM.block(0, 3, 4, 3) = v_bands[i];

            mean = BDM.rowwise().mean();
            for(int a = 0; a < 6; a++) {
                deviation = deviation + ((BDM.col(a) - mean).array() * (BDM.col(a) - mean).array()).matrix();
            }
            deviation = (deviation/5).array().sqrt().matrix();
        }
        if(i > 0 && i < bands.size() - 1) {
            Eigen::Matrix<double, 4, 9> BDM;
            BDM.block(0, 0, 4, 3) = v_bands[i - 1];
            BDM.block(0, 3, 4, 3) = v_bands[i];
            BDM.block(0, 6, 4, 3) = v_bands[i + 1];

            mean = BDM.rowwise().mean();
            for(int a = 0; a < 9; a++) {
                deviation = deviation + ((BDM.col(a) - mean).array() * (BDM.col(a) - mean).array()).matrix();
            }
            deviation = (deviation/8).array().sqrt().matrix();
        }
        
        // 将描述子写入LBD中
        LBD.block(0, i, 4, 1) = mean;
        LBD.block(4, i, 4, 1) = deviation;
    }

    

#ifdef Debug
    std::cout << "LBD: \n" << LBD << std::endl;
    for(int i = 0; i < v_bands.size(); i++) {
        std::cout << "v_b " << i << std::endl;
        std::cout << v_bands[i] << std::endl;
    }
#endif

    return LBD;

}

Matrix43 LSR::computeDescriptor_singleBand(Band &band) {
    Eigen::MatrixXd v_b(4, 3);
    double v1, v2, v3, v4;
    int rows = band.gx.rows;
    int cols = band.gy.cols;
    for(int i = 0; i < rows; i++) {
        double lambda = band.global_weight[i] * band.local_weight[i];
        v1 = 0;
        v2 = 0;
        v3 = 0;
        v4 = 0;
        for(int j = 0; j < cols; j++) {
            if(band.gx.at<float>(i, j) > 0) {
                v1 = v1 + band.gx.at<float>(i, j); 
            }
            if(band.gx.at<float>(i, j) <= 0) {
                v2 = v2 - band.gx.at<float>(i, j);
            }
            if(band.gy.at<float>(i, j) > 0) {
                v3 = v3 + band.gy.at<float>(i, j); 
            }
            if(band.gy.at<float>(i, j) <= 0) {
                v4 = v4 - band.gy.at<float>(i, j);
            }
            
        }
        v1 = lambda * v1;
        v2 = lambda * v2;
        v3 = lambda * v3;
        v4 = lambda * v4;
        v_b(0, i) = v1;
        v_b(1, i) = v2;
        v_b(2, i) = v3;
        v_b(3, i) = v4;
    }

    return v_b;
}

void LSR::showGradient() {
    cv::imshow("Band 1 dx", band1_.gx);
    cv::imshow("Band 1 dy", band1_.gy);

    cv::imshow("Band 2 dx", band2_.gx);
    cv::imshow("Band 2 dy", band2_.gy);

    cv::imshow("Band 3 dx", band3_.gx);
    cv::imshow("Band 3 dy", band3_.gy);

    cv::imshow("Band 4 dx", band4_.gx);
    cv::imshow("Band 4 dy", band4_.gy);

    cv::imshow("Band 5 dx", band5_.gx);
    cv::imshow("Band 5 dy", band5_.gy);

    cv::waitKey(0);
}

MatrixBD& LSR::returnDescriptor() {
    return LBD_;
}