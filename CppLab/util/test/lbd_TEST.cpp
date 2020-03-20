//
// Author: LHO LHospitalLKY@github.com 
//

#include <chrono>

#include "../LBD.h"
#include "../Gradient.h"
#include "../line_tracking.h"

void showLSD(); // 展示line_descriptor中的lsd线特征计算会得到什么数据
void flowMatch();
void LineDescriptorCal();

int main(int argc, char *argv[]) {
    showLSD();
    cv::waitKey(0);
    LineDescriptorCal();
    cv::waitKey(0);
    flowMatch();
}

void showLSD() {
    // cv::Mat frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/data_stereo_flow_multiview/testing/image_2/000000_00.png");
    cv::Mat frame_0 = cv::imread("/home/lho/SLAM/FlowNet/Flow_lab/Line Segment Detector/data/LK1.png");    
    cv::Mat frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638518227829504.png");

    cv::Mat frame_0_tmp;
    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    // frame_0_tmp.convertTo(frame_0_gray, CV_32FC1, 1/255.0);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);

    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
    std::vector<cv::line_descriptor::KeyLine> lines;
    cv::Mat descriptor;
    lsd -> detect(frame_0, lines, 5, 1, cv::Mat());

    std::cout << lines.size() << std::endl;

    for(auto &line : lines) {
        std::cout << "=================" << std::endl;
        std::cout << "class id: " << line.class_id << std::endl;
        std::cout << "Octave: " << line.octave << std::endl;
        std::cout << "Angle: " << line.angle << std::endl;
        std::cout << "Start point: " << line.getStartPoint() << std::endl;
        std::cout << "Start point in octave: " << line.getStartPoint() << std::endl;
        std::cout << "End point in octave: " << line.getEndPointInOctave() << std::endl;
        std::cout << "End point: " << line.getEndPoint() << std::endl; 
        std::cout << "Center: " << line.pt << std::endl;
        std::cout << "Line Length: " << (int)line.lineLength << std::endl;
        std::cout << "Line Size: " << line.size << std::endl;
    }
}

void LineDescriptorCal() {

    // 读取图像
    cv::Mat img = cv::imread("/home/lho/SLAM/FlowNet/Flow_lab/Line Segment Detector/data/LK1.png", CV_8UC1);
    // 提取线段
    std::vector<cv::line_descriptor::KeyLine> lines;
    cv::Ptr<cv::line_descriptor::LSDDetector> lsd = cv::line_descriptor::LSDDetector::createLSDDetector();
    lsd -> detect(img, lines, 5, 1, cv::Mat());

    auto start_time = std::chrono::system_clock::now();
    GradientMap gm;
    std::shared_ptr<Gradient> gradient = std::make_shared<Gradient>();
    gradient -> compute(img);
    gm = gradient -> returnGradient();

    std::vector<MatrixBD, Eigen::aligned_allocator<MatrixBD>> descriptors(lines.size());
#pragma omp parallel for 
    for(int i = 0; i < lines.size(); i++) {
        std::shared_ptr<LSR> lsr = std::make_shared<LSR>();
        lsr -> computeBands(img, lines[i], gm);
        descriptors[i] = lsr -> returnDescriptor();
    }

    auto end_time = std::chrono::system_clock::now();

    auto time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "My time uesd: " << time_used.count() << "ms" << std::endl;

    std::shared_ptr<LSR> lsr = std::make_shared<LSR>();
    lsr -> computeBands(img, lines[10], gm);
    lsr -> showGradient();
    MatrixBD line10_BD = lsr -> returnDescriptor();

    std::cout << "LINE 10 BD: \n" << line10_BD << std::endl;
    std::cout << "LINES[10]: \n" << descriptors[10] << std::endl;
    
}

void flowMatch() {
    // 读取两张图片
    cv::Mat frame_0, frame_1;
    frame_0 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638522777829376.png");
    frame_1 = cv::imread("/media/lho/064A027D79FA99C7/slam数据集/SLAM数据集/EuRoC/MH_05_difficult/mav0/cam0/data/1403638522877829376.png");

    cv::Mat frame_0_gray, frame_1_gray;
    cv::cvtColor(frame_0, frame_0_gray, CV_BGR2GRAY);
    cv::cvtColor(frame_1, frame_1_gray, CV_BGR2GRAY);

    // 计算直线
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(LSD_REFINE_ADV);
    std::vector<cv::Vec4f> keylines;
    lsd -> detect(frame_0_gray, keylines);
    cv::Mat frame_0_lines(frame_0);
    lsd -> drawSegments(frame_0_lines, keylines);
    cv::imshow("frame 0 lines", frame_0_lines);

    // 计算直线的描述子
    auto start_Grad = std::chrono::system_clock::now();
    // 计算梯度
    GradientMap gm_frame_0, gm_frame_1;
    std::shared_ptr<Gradient> gradient_frame_0 = std::make_shared<Gradient>();
    std::shared_ptr<Gradient> gradient_frame_1 = std::make_shared<Gradient>();
    // 图像0的梯度
    gradient_frame_0 -> compute(frame_0_gray);
    gm_frame_0 = gradient_frame_0 -> returnGradient();
    // 图像1的梯度
    gradient_frame_1 -> compute(frame_1_gray);
    gm_frame_1 = gradient_frame_1 -> returnGradient();
    auto end_Grad = std::chrono::system_clock::now();
    auto Grad_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_Grad - start_Grad);
    std::cout << "Grad time uesd: " << Grad_time_used.count() << "ms" << std::endl;

    
    // 计算光流
    cv::Mat flow;
    cv::Ptr<cv::DenseOpticalFlow> deep_flow = cv::optflow::createOptFlow_DeepFlow();
    deep_flow -> calc(frame_0_gray, frame_1_gray, flow);
    // 计算直线流动
    std::vector<cv::Vec4f> keylines_frame_1;
    for(auto &line : keylines) {
        cv::Point2f start, end, center;
        start = cv::Point2f(line[0], line[1]);
        end = cv::Point2f(line[2], line[3]);
        center = 0.5 * (start + end);
        float lineLength = std::sqrt(std::pow(start.x - end.x, 2) + std::pow(start.y - end.y, 2));
        
        // 起点的流动
        cv::Point2f start_esti;
        start_esti = start + flow.at<cv::Point2f>(start);
        // 重点的流动
        cv::Point2f end_esti;
        end_esti = end + flow.at<cv::Point2f>(end);
        // 中点的流动
        cv::Point2f center_esti;
        center_esti = center + flow.at<cv::Point2f>(center);

        // 长度
        float lineLength_esti;
        lineLength_esti = std::sqrt(std::pow(start_esti.x - end_esti.x, 2) + std::pow(start_esti.y - end_esti.y, 2));

        cv::Vec4f line_esti;
        line_esti[0] = start_esti.x;
        line_esti[1] = start_esti.y;
        line_esti[2] = end_esti.x;
        line_esti[3] = end_esti.y;
        // TODO: 加判断
        keylines_frame_1.push_back(line_esti);        

        std::cout << "==========================" << std::endl;
        std::cout << "Start: " << start << std::endl;
        std::cout << "Start Estim: " << start_esti << std::endl;
        std::cout << "End: " << end << std::endl;
        std::cout << "End Estim: " << end_esti << std::endl;
        std::cout << "Center: " << center << std::endl;
        std::cout << "Center Estim: " << center_esti << std::endl;
        std::cout << "Length: " << lineLength << std::endl;
        std::cout << "Length Estim: " << lineLength_esti << std::endl;
    }

    // cv::Mat frame_1_lines(frame_1);
    // lsd -> drawSegments(frame_1_lines, keylines_frame_1);
    // cv::imshow("frame 1 esti", frame_1_lines);

    // 计算线段特征描述
    // 计算frame_0线特征的描述子
    auto start_BD = std::chrono::system_clock::now();
    std::vector<MatrixBD, Eigen::aligned_allocator<MatrixBD>> descriptor_frame_0(keylines.size());
    #pragma omp parallel for 
    for(int i = 0; i < keylines.size(); i++) {
        std::shared_ptr<LSR> lsr = std::make_shared<LSR>();
        lsr -> computeBands(frame_0, keylines[i], gm_frame_0);
        descriptor_frame_0[i] = lsr -> returnDescriptor();
    }
    // 计算frame_1线特征的描述子
    std::vector<MatrixBD, Eigen::aligned_allocator<MatrixBD>> descriptor_frame_1(keylines_frame_1.size());
    #pragma omp parallel for 
    for(int i = 0; i < keylines_frame_1.size(); i++) {
        std::shared_ptr<LSR> lsr = std::make_shared<LSR>();
        lsr -> computeBands(frame_1, keylines_frame_1[i], gm_frame_1);
        descriptor_frame_1[i] = lsr -> returnDescriptor();
    }
    auto end_BD = std::chrono::system_clock::now();
    auto BD_time_used = std::chrono::duration_cast<std::chrono::milliseconds>(end_BD - start_BD);
    std::cout << "BD time uesd: " << BD_time_used.count() << "ms" << std::endl;

     
    std::cout << ">>>>>>>>>>>>>>>>>>>" << std::endl;
    std::cout << "Descriptor 0 size: " << descriptor_frame_0.size() << std::endl;
    std::cout << "Descriptor 1 size: " << descriptor_frame_1.size() << std::endl;
    std::cout << ">>>>>>>>>>>>>>>>>>>" << std::endl;

    // 删去不匹配的特征
    std::vector<cv::Vec4f> lines_tracked;
    for(int i = 0; i < descriptor_frame_0.size(); i++) {
        MatrixBD desc_0, desc_1;
        desc_0 = descriptor_frame_0[i];
        desc_1 = descriptor_frame_1[i];
        double err;
        err = (desc_0 - desc_1).mean();
        if(std::fabs(err) < 0.6) {
           lines_tracked.push_back(keylines_frame_1[i]);
        }
    }

    std::cout << lines_tracked.size() << std::endl;

    cv::Mat frame_1_line_tracked(frame_1);
    lsd -> drawSegments(frame_1_line_tracked, lines_tracked);
    cv::imshow("frame 1 lines", frame_1_line_tracked);

    cv::waitKey(0);
}

