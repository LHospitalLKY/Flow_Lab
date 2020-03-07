//
// Author: LHO LHospitalLKY@github.com 
//

#include "read_write.h"

void write(std::string fileName, cv::Mat &flo) {

    // 获取flo的size
    cv::Size imgsize = flo.size();
	int height = imgsize.height;
	int width = imgsize.width;

    // 写入文件
	std::ofstream fout(fileName, std::ios::binary);
	char * data = flo.ptr<char>(0);
	if (!fout) { 
		std::cerr << "Wrong of writing flo!" << std::endl;
		return; 
	}
	else {
		fout << "PIEH";
		fout.write((char*)&width, sizeof(int));
		fout.write((char*)&height, sizeof(int));
		fout.write(data, height * width * 2 * sizeof(float));
	}
	fout.close();
}

void read(std::string fileName, cv::Mat &flo) {
    
    // 打开文件，读取数据
    std::ifstream fin(fileName, std::ios::binary);
	char buffer[sizeof(float)];
	fin.read(buffer, sizeof(float));
    // tar是什么
	float tar = ((float*)buffer)[0];
	if (tar != TAR) {
		fin.close();
		return;
	}

    // 读取flo的size
	fin.read(buffer, sizeof(int));
	int height = ((int*)buffer)[0];
	fin.read(buffer, sizeof(int));
	int width = ((int*)buffer)[0];
	flo = cv::Mat(cv::Size(height, width), CV_32FC2);
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < height; j++) {
			if (!fin.eof()) {
				float * data = flo.ptr<float>(i, j);
				fin.read(buffer, sizeof(float));
				data[0] = ((float*)buffer)[0];
				fin.read(buffer, sizeof(float));
				data[1] = ((float*)buffer)[0];
			}	
		}
	}
	fin.close();

}