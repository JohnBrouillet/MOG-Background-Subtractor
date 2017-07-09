#ifndef MGG_SUBSTRACTOR_H
#define MGG_SUBSTRACTOR_H

#include "opencv_include.h"
#include <iostream>
#include <chrono>
#include <omp.h>

using namespace cv;
using namespace cv::ml;

/* Threshold value for matching */
const float k = 2.5;

/* Minimum covariance for a gaussian. Set to avoid division by 0. */
const float cov_min = 2.0;

class MOGBackgroundSubtraction{

public:
	/* According to Stauffer and Grimson, K should be set from 3 to 5. 
	 K represents the number of gaussians per pixel that model the background and the foreground.
	 ratio parameter can be set to resize the image in order to increase (or decrease) FPS.
	 ratio > 1 : image size is reduced 
	 0 < ratio < 1 : image size is increased 
	 a is the learning rate.
	 T is the minimal weight for a gaussian distribution to be considered as a background 	   */
	MOGBackgroundSubtraction(int _K = 3, float _a = 0.9, float _T = 0.5, int _downsample = 1);

	/* Initialize the W*H*K gaussians. */
	void init(std::vector<Mat>& imgs);

	/* Create the mask. Black for the background, white for the foreground. */
	Mat createMask(Mat& img);

private:
	/* Create a Mat structure whose columns are formed from the images in the vector */
	void zipStdVecToMat(std::vector<Mat>& imgs, Mat& output);

	/* Apply the expectation maximization algorithm to initialize the gaussians */
	void train(Mat& line, int id_line, Mat& _cov, Mat& _mean, Mat& _weight);

	/* Resize and convert in grayscale the image */
	void wrapTransform(Mat& input);

	/* Check if the pixels match a gaussian */
	void isInGaussian(Mat& X, Mat& maha, Mat& mask_own);

	/* Compute the gaussian probability density function for each pixels */
	void computeGaussianProbDensity(Mat& maha, Mat& probDensity);

	/* Create the mask */
	void masking(Mat& X, Mat& mask, Mat& mask_own, Mat& least_prob, Mat& prob);

	/* Update the gaussians for the case : a match is found with a gaussian */
	void update_case1(uchar X, int idx_match, Mat& prob, Mat& _cov, Mat& _mean, Mat& _weight);

	/* Update the gaussians for the case : no match is found with a gaussian */
	void update_case2(uchar X, Mat& least_prob, Mat& _cov, Mat& _mean, Mat& _weight);

	/* Apply morphological operations to filter noise and refine the blobs */
	void morphoOp(Mat& mask);

	int nb_gauss;
	int downsample;
	int WH;

	float a;
	float T;

	Mat cov, mean, weight, element3, element5;

};


#endif