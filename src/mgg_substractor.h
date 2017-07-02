#ifndef MGG_SUBSTRACTOR_H
#define MGG_SUBSTRACTOR_H

#include "opencv_include.h"
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace cv::ml;

const float k = 2.5;
const float a = 0.9;
const float T = 0.5;
const float sig_min = 2.0;

class MOGBackgroundSubtraction{

public:
	MOGBackgroundSubtraction(int _N = 3, int _ratio = 1);
	void init(std::vector<Mat>& imgs);
	void createMask(Mat& X, Mat& mask);

private:
	void zipStdVecToMat(std::vector<Mat>& imgs, Mat& output);
	void train(Mat& line, int id_line, Mat& _cov, Mat& _mean, Mat& _weight);
	void wrapTransform(Mat& input);
	void isInGaussian(Mat& X, Mat& maha, Mat& mask_own);
	void computeGaussianProbDensity(Mat& maha, Mat& probDensity);
	void masking(Mat& X, Mat& mask, Mat& mask_own, Mat& least_prob, Mat& prob);
	void update_case1(uchar X, int idx_match, Mat& prob, Mat& _cov, Mat& _mean, Mat& _weight);
	void update_case2(uchar X, Mat& least_prob, Mat& _cov, Mat& _mean, Mat& _weight);

	int nb_gauss;
	int ratio;
	int WH;

	Mat cov, mean, weight;

};


#endif