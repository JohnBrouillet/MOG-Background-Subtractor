#ifndef MGG_SUBSTRACTOR_H
#define MGG_SUBSTRACTOR_H

#include "opencv_include.h"
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace cv::ml;

class MGGBackgroundSubstractor{

public:
	MGGBackgroundSubstractor(int _N = 3, int _ratio = 1);
	void init(std::vector<Mat>& imgs);

private:
	void zipStdVecToMat(std::vector<Mat>& imgs, Mat& output);
	void train(Mat& line, int id_line, Mat& _cov, Mat& _mean, Mat& _weight);

	int nb_gauss;
	int ratio;

	Mat cov, mean, weight;

};


#endif