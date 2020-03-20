//
// Author: LHO LHospitalLKY@github.com 
//

#include "feature_tracker.h"

#define MIN_LINES 1000

void reduceVector(vector<cv::Vec4f> &v, vector<uchar> status) {
    if(v.empty()) {
        return;
    }
    int j = 0;
    for(int i = 0; i < status.size(); i++) {
        if(status[i]) {
            v[j] = v[i];
            j++;
        }
    }
    v.resize(j);
}
void reduceVector(vector<int> &v, vector<uchar> status) {
    if(v.empty()) {
        return;
    }
    int j = 0;
    for(int i = 0; i < status.size(); i++) {
        if(status[i]) {
            v[j] = v[i];
            j++;
        }
    }
    v.resize(j);
}

FeatureTracker::FeatureTracker() {}

void FeatureTracker::readIntrinsicParameter(const cv::Mat &K, const std::vector<double> &dist) {
    K_ = K;
    dist_ = dist;
}

// TODO: 补充时间计算
// TODO: 补充注释
// TODO: 补充CLAHE处理符号
void FeatureTracker::readImage(const cv::Mat &img, double cur_time, cv::Mat &drawLines, cv::Mat &flow) {
    cv::Mat image;

    // 检查图片是否是CV_8UC1格式
    // TODO: 改为其他方法的时候，可能会要求RGB格式，这里先这么写，之后改一下
    if(img.type() != CV_8UC1) {
        std::cerr << "Image is not CV_8UC1" << std::endl;
    }
    // 填充图像
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe -> apply(img, image);

    if(forw_img_.empty()) {
        // 若forw_img_为空，说明是首次接收到图像
        prev_img_ = cur_img_ = forw_img_ = image;
    } else {
        // 否则，只更新forw_img_
        forw_img_ = image;
    }

    std::shared_ptr<LineOpticalFlow> lof = std::make_shared<LineOpticalFlow>();

    // 跟踪线特征
    forw_lines_.clear();
    if(cur_lines_.size() > 0) {

        vector<uchar> status;
        cv::Mat flow_color;
        flow_color = lof -> calc(cur_img_, forw_img_, cur_lines_, forw_lines_, status);
        flow = flow_color;
        cv::imshow("flow", flow_color);

        for(int i = 0; i < forw_lines_.size(); i++) {
            if(status[i] && !inBorder(forw_lines_[i])) {
                status[i] = 0;
            }
        }

        reduceVector(prev_lines_, status);
        reduceVector(cur_lines_, status);
        reduceVector(forw_lines_, status);

        std::cout << "==========At Start===========" << std::endl;
        std::cout << "forw_lines size: " << forw_lines_.size() << std::endl;
        std::cout << "cur_lines size: " << cur_lines_.size() << std::endl;
        std::cout << "prev_lines size: " << prev_lines_.size() << std::endl;

        // TODO: 调试结束之后删去
        // std::cout << "------------------" << std::endl;
        // std::cout << "num of cur_lines:" << cur_lines_.size() << std::endl;
        // std::cout << "------------------" << std::endl;
        
    }

    // TODO: 完善PUB_THIS_FRAME指数
    // TODO: 补充最大数量限制
    rejectWithF();

    std::cout << "---------At reject---------" << std::endl;
    std::cout << "forw_lines size: " << forw_lines_.size() << std::endl;
    std::cout << "cur_lines size: " << cur_lines_.size() << std::endl;
    std::cout << "prev_lines size: " << prev_lines_.size() << std::endl;

    setMask();
    // 寻找新的线特征
    if(mask_.empty()) {
        std::cout << "mask is empty" << std::endl;
    }
    if(mask_.type() != CV_8UC1) {
        std::cout << "mask type wrong" << std::endl;
    }
    if(mask_.size() != forw_img_.size()) {
        std::cout << "mask wrong size" << std::endl;
    }
    lineDetect(forw_img_, n_lines_, mask_);
    std::cout << "num of n_lines " << n_lines_.size() << std::endl;
    std::cout << "-----------------" << std::endl;

    addLines();

    // cv::Mat drawLines;
    drawLines = lof -> drawTracking(forw_img_, forw_lines_);
    cv::imshow("lines", drawLines);
    cv::waitKey(10);

    // 更新图像和线特征
    prev_img_ = cur_img_;
    prev_lines_ = cur_lines_;
    cur_img_ = forw_img_;
    cur_lines_ = forw_lines_;

    prev_time_ = cur_time_;

}

// Mat.at<>(i, j)参数中，先row后col
// Vec4f中，是先x后y，x表示col，y表示row

bool FeatureTracker::inBorder(const cv::Vec4f &line) {
    double start_x, start_y;
    double end_x, end_y;
    start_x = line[0];
    start_y = line[1];
    end_x = line[2];
    end_y = line[3];
    if(start_x < 0 || start_x > forw_img_.cols)
        return false;
    if(start_y < 0 || start_y > forw_img_.rows)
        return false;
    if(end_x < 0 || end_x > forw_img_.cols)
        return false;
    if(end_y < 0 || end_y > forw_img_.cols)
        return false;

    return true;
}

/**
 * @brief: 为forw_img_设置mask，使得forw_img_中找出的线特征与跟踪到的线特征不重复
 * @param: 
 * @return: 
 * @ret: 
 **/
void FeatureTracker::setMask() {
    // TODO: 设置fisheye的mask
    // TODO: 补充完整
    mask_ = cv::Mat(forw_img_.size(), CV_8UC1, cv::Scalar(255));
    if(forw_lines_.empty()) {
        return;
    }
    for(int i = 0; i < forw_lines_.size(); i++) {
        cv::Point2f start, end;
        start.x = forw_lines_[i][0];
        start.y = forw_lines_[i][1];
        end.x = forw_lines_[i][2];
        end.y = forw_lines_[i][3];
        cv::circle(mask_, start, 5, 0, -1);
        cv::circle(mask_, end, 5, 0, -1);
    }

}

// TODO: 完善内参部分
void FeatureTracker::rejectWithF() {

    if(forw_lines_.empty()) {
        return;
    }

    // 内参变换

    vector<uchar> status_s, status_e, status;
    // 将线段拆成两个队列
    assert(cur_lines_.size() == forw_lines_.size());
    vector<cv::Point2f> cur_starts, cur_ends;
    vector<cv::Point2f> forw_starts, forw_ends;

    vector<cv::Point2f> und_cur_starts, und_cur_ends;
    vector<cv::Point2f> und_forw_starts, und_forw_ends;

/*
    for(int i = 0; i < cur_lines_.size(); i++) {
        cur_starts.push_back(cv::Point2f(cur_lines_[i][0], cur_lines_[i][1]));
        cur_ends.push_back(cv::Point2f(cur_lines_[i][2], cur_lines_[i][3]));
        forw_starts.push_back(cv::Point2f(forw_lines_[i][0], forw_lines_[i][1]));
        forw_ends.push_back(cv::Point2f(forw_lines_[i][2], forw_lines_[i][3]));
    }
*/
    for(int i = 0; i < cur_lines_.size(); i++) {
        cur_starts.push_back(cv::Point2f(cur_lines_[i][0], cur_lines_[i][1]));
        cur_ends.push_back(cv::Point2f(cur_lines_[i][2], cur_lines_[i][3]));
        forw_starts.push_back(cv::Point2f(forw_lines_[i][0], forw_lines_[i][1]));
        forw_ends.push_back(cv::Point2f(forw_lines_[i][2], forw_lines_[i][3]));
    }

    cv::fisheye::distortPoints(cur_starts, und_cur_starts, K_, dist_);
    cv::fisheye::distortPoints(cur_ends, und_cur_ends, K_, dist_);
    cv::fisheye::distortPoints(forw_starts, und_forw_starts, K_, dist_);
    cv::fisheye::distortPoints(forw_ends, und_forw_ends, K_, dist_);

    // TODO: 第四个参数要进行自定义
    cv::findFundamentalMat(und_cur_starts, und_forw_starts, cv::FM_RANSAC, 10, 0.99, status_s);
    cv::findFundamentalMat(und_cur_ends, und_forw_ends, cv::FM_RANSAC, 10, 0.99, status_e);

    int count_valid = 0;
    for(int i = 0; i < status_e.size(); i++) {
        if(status_e[i] != 0 && status_s[i] != 0) {
            status.push_back(1);
            count_valid++;
        } else {
            status.push_back(0);
        }
    }

    std::cout << "Num of status in RejectF: " << status.size() << std::endl;

    reduceVector(prev_lines_, status);
    reduceVector(cur_lines_, status);
    reduceVector(forw_lines_, status);

    return;

}

// TODO: 增加id与跟踪次数
void FeatureTracker::addLines() {
    for(auto &l : n_lines_) {
        forw_lines_.push_back(l);
    }
}


void FeatureTracker::lineDetect(const cv::Mat &img, vector<cv::Vec4f> &lines, const cv::Mat &mask) {
    lines.clear();
    // std::cout << "================================" << std::endl;
    // std::cout << "forw_lines_ num in lineDetect()" << forw_lines_.size() << std::endl;
    // std::cout << "n_lines_ num in lineDetect()" << n_lines_.size() << std::endl;
    if(forw_lines_.size() > MIN_LINES) {
        std::cout << "forw lines size large than min_lines" << std::endl;
        return;
    }

    std::vector<cv::Vec4f> lines_tmp;
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(LSD_REFINE_STD);
    lsd -> detect(img, lines_tmp);

    if(lines_tmp.empty()) {
        std::cout << "Not found Lines!" << std::endl;
        return;
    }
    
    for(int i = 0; i < lines_tmp.size(); i++) {
        cv::Point2f start, end;
        start.x = lines_tmp[i][0];
        start.y = lines_tmp[i][1];
        end.x = lines_tmp[i][2];
        end.y = lines_tmp[i][3];
        if(inBorder(lines_tmp[i]) && mask.at<uchar>(start.y, start.x) == 0) {
            continue;
        }
        if(inBorder(lines_tmp[i]) && mask.at<uchar>(end.y, end.x) == 0) {
            continue;
        }
        lines.push_back(lines_tmp[i]);
    }
}

