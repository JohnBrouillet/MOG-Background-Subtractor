#include "mgg_substractor.h"

MGGBackgroundSubstractor::MGGBackgroundSubstractor(int _N, int _ratio)
{
	nb_gauss = _N;
	ratio = _ratio;
}

void MGGBackgroundSubstractor::init(std::vector<Mat>& imgs)
{
	for(Mat& img : imgs)
    	cv::resize(img, img, cv::Size(), 1.0/ratio, 1.0/ratio);
    
	std::cout << "Initialization" << std::endl;
	Mat cube;
	zipStdVecToMat(imgs, cube);

	int WH = imgs[0].rows*imgs[0].cols;
	Mat _cov = Mat::zeros(WH, nb_gauss, CV_64F);
	Mat _mean = Mat::zeros(WH, nb_gauss, CV_64F); 
	Mat _weight = Mat::zeros(WH, nb_gauss, CV_64F);

	int i;
	auto t1 = std::chrono::high_resolution_clock::now();
	#pragma omp parallel for shared(_cov, _mean, _weight), private(i)
	for(i = 0; i < WH; i++)
	{
		Mat tmp = cv::Mat::zeros(1, cube.cols, CV_8UC1);
		cube.row(i).copyTo(tmp);
		train(tmp, i, _cov, _mean, _weight);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << WH << " EM algo performed in " << fp_ms.count() << "ms" << std::endl;
}

void MGGBackgroundSubstractor::zipStdVecToMat(std::vector<Mat>& imgs, Mat& output)
{
	int WH = imgs[0].rows*imgs[0].cols;
	Mat cube = Mat::zeros(imgs.size(), WH, CV_8UC1); 

	int idx = 0;
	for(const cv::Mat& img : imgs)
	{
		Mat tmp(1, WH, CV_8UC1, img.data);
		tmp.copyTo(cube.row(idx));
		++idx;
	}
	output = cube.t();

}

void MGGBackgroundSubstractor::train(Mat& line, int id_line, Mat& _cov, Mat& _mean, Mat& _weight)
{	
    Ptr<EM> model = EM::create();
    model->setClustersNumber(nb_gauss);
    model->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
    model->setTermCriteria(TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 0.1));
    model->trainEM( line.t() );

	Mat m = model->getMeans().t();
	std::vector<Mat> c; model->getCovs(c);
	Mat w = model->getWeights();

	m.copyTo(_mean.row(id_line));
	w.copyTo(_weight.row(id_line));
	hconcat(c, _cov.row(id_line));
}