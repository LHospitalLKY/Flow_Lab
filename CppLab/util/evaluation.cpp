//
// Author: LHO LHospitalLKY@github.com 
//

#include "evaluation.h"

double endPointError(cv::Mat &estimFlow, cv::Mat &gtFlow, cv::Mat &EPE_Mat) {
    // 判断二者的大小
    assert(estimFlow.cols == gtFlow.cols);
    assert(estimFlow.rows == gtFlow.rows);
    assert(estimFlow.channels() == 2);
    assert(gtFlow.channels() == 2);

    int rows = estimFlow.rows;
    int cols = estimFlow.cols;
    // cv::Mat epeMap;
    std::vector<cv::Mat> estFlow_uv;
    std::vector<cv::Mat> gtFlow_uv;
/*
    Eigen::ArrayXd estFlow_u(rows, cols);
    Eigen::ArrayXd estFlow_v(rows, cols);
    Eigen::ArrayXd gtFlow_v(rows, cols);
    Eigen::ArrayXd gtFlow_v(rows, cols);
*/

    cv::split(estimFlow, estFlow_uv);
    cv::split(gtFlow, gtFlow_uv);
    
    double sum_err = 0;
    for(int h = 0; h < rows; h++) {
        for(int w = 0; w < cols; w++) {
            
            EPE_Mat.at<float>(h, w) = 0;
            float u_err = std::fabs(estFlow_uv[0].at<float>(h, w) - gtFlow_uv[0].at<float>(h, w));
            float v_err = std::fabs(estFlow_uv[1].at<float>(h, w) - gtFlow_uv[1].at<float>(h, w));

            double err = std::sqrt(std::pow(u_err, 2) + std::pow(v_err, 2));

            EPE_Mat.at<float>(h, w) = err;
            sum_err = sum_err + err;
        }
    }

    return sum_err / (cols * rows);

}
