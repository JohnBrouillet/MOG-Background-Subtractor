#include "mgg_substractor.h"

MOGBackgroundSubtraction::MOGBackgroundSubtraction(int _N, int _ratio)
{
	nb_gauss = _N;
	ratio = _ratio;
}

void MOGBackgroundSubtraction::wrapTransform(Mat& input)
{
	cv::resize(input, input, cv::Size(), 1.0/ratio, 1.0/ratio);
	cv::cvtColor(input, input, CV_BGR2GRAY);
}

void MOGBackgroundSubtraction::init(std::vector<Mat>& imgs)
{
	for(Mat& img : imgs)
    	wrapTransform(img);

    WH = imgs[0].rows*imgs[0].cols;
	std::cout << "Initialization" << std::endl;
	Mat cube;
	zipStdVecToMat(imgs, cube);

	
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

    cov = _cov;
    mean = _mean;
    weight = _weight;
}

void MOGBackgroundSubtraction::createMask(Mat& X, Mat& mask)
{
	wrapTransform(X);

	Mat maha, mask_own, prob, least_prob;
	divide(weight, cov, least_prob);
	isInGaussian(X, maha, mask_own);
	computeGaussianProbDensity(maha, prob);
	masking(X, mask, mask_own, least_prob);

}

void MOGBackgroundSubtraction::zipStdVecToMat(std::vector<Mat>& imgs, Mat& output)
{
	Mat cube = Mat::zeros(imgs.size(), WH, CV_8UC1); 

	for(int idx = 0; idx < imgs.size(); idx++)
	{
		Mat tmp(1, WH, CV_8UC1, imgs[idx].data);
		tmp.copyTo(cube.row(idx));
	}
	output = cube.t();

}

void MOGBackgroundSubtraction::train(Mat& line, int id_line, Mat& _cov, Mat& _mean, Mat& _weight)
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

void MOGBackgroundSubtraction::isInGaussian(Mat& X, Mat& maha, Mat& mask_own)
{	
	Mat in = Mat(WH,1, X.type(), X.data);
	Mat out;
	repeat(in, 1, nb_gauss, out);

	out.convertTo(out, CV_64F);
	Mat tmp = out-mean;
	divide(tmp, cov, maha);

	mask_own = maha < k; //if 255, the gaussian is matched
	multiply(maha, tmp, maha);
}

void MOGBackgroundSubtraction::computeGaussianProbDensity(Mat& maha, Mat& probDensity)
{
	probDensity = -0.5*maha;
	exp(probDensity, probDensity);
	Mat sqrt_cov;
	sqrt(cov, sqrt_cov);
	divide(a*probDensity, 2*CV_PI*(sqrt_cov), probDensity);
}

void MOGBackgroundSubtraction::masking(Mat& X, Mat& mask, Mat& mask_own, Mat& least_prob)
{
	Mat what_case;
	reduce(mask_own, what_case, 1, CV_REDUCE_SUM, CV_32SC1); 
	what_case = what_case > 0; // if 255, a gaussian is matched

	//For OpenMP
	Mat _cov = cov;
	Mat _mean = mean;
	Mat _weight = weight;

	uchar* case_data = what_case.data;
	uchar* mask_data = mask.data;
	uchar* pixel = X.data;
	int i;
	for(i = 0; i < WH; i++)
	{
		uchar cas = case_data[i];
		uchar pix = pixel[i];
		if(cas)
		{

		}
		else
		{
			mask_data[i] = 255;	
			Mat tmp_l = least_prob.row(i), tmp_c = _cov.row(i), tmp_m = _mean.row(i), tmp_w = _weight.row(i);
			maj_case2(pix, tmp_l, tmp_c, tmp_m, tmp_w);
		}
	}
	cov = _cov;
	mean = _mean;
	weight = _weight;
}

void MOGBackgroundSubtraction::maj_case1()
{


}

void MOGBackgroundSubtraction::maj_case2(uchar X, Mat& least_prob, Mat& _cov, Mat& _mean, Mat& _weight)
{
	Mat idx;
	sortIdx(least_prob, idx, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);
	int least = idx.data[0];

	_cov.at<double>(0,least) = 20;
	_mean.at<double>(0, least) = (double)X;
	_weight.at<double>(0, least) = 0.1;
}