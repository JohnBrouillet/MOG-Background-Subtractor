#include <iostream>
#include "utils.h"
#include "mgg_substractor.h"

const int nb_frame_init = 50;
const int nb_gauss = 3;
const int downsample = 2;

void cam_loop(cv::VideoCapture& cap)
{
	std::vector<cv::Mat> data;
	for(int i = 0; i < nb_frame_init; i++)
	{
		cv::Mat img;
		cap >> img;

		data.push_back(img);
	}

	MOGBackgroundSubtraction mg(nb_gauss, downsample);
	mg.init(data);

	int count = 0;
	auto t1 = std::chrono::high_resolution_clock::now();
	while(1)
	{
		
		cv::Mat img;
		cap >> img;
		Mat mask = Mat::zeros(img.rows/downsample, img.cols/downsample, CV_8UC1);
		
		mg.createMask(img, mask);
	
		cv::imshow("MyWindow", mask);
		if (cv::waitKey(30) == 27) 
   		{
       	 	std::cout << "esc key is pressed by user" << std::endl;
        	break; 
   		}
		
   		count++; 
		if(count == 30)
		{
			auto t2 = std::chrono::high_resolution_clock::now();
		    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
		    std::cout << "FPS " << count / (fp_ms.count()*1e-3) << std::endl;
		    count = 0;
		    t1 = std::chrono::high_resolution_clock::now();
		}
	}
}


void image(std::string path)
{
	cv::namedWindow("mask", CV_WINDOW_AUTOSIZE);
	cv::namedWindow("img", CV_WINDOW_AUTOSIZE);
	std::vector<std::string> files = open_files(path);
	std::vector<cv::Mat> data;
	for(int i = 0; i < nb_frame_init; i++)
	{
		cv::Mat img = cv::imread(std::string(path) + files[i], CV_LOAD_IMAGE_COLOR);
		data.push_back(img);
	}

	MOGBackgroundSubtraction mg(nb_gauss,downsample);
	mg.init(data);

	for(int i = nb_frame_init; i < files.size(); i++)
	{
		cv::Mat img = cv::imread(std::string(path) + files[i], CV_LOAD_IMAGE_COLOR);
		Mat mask = Mat::zeros(img.rows/downsample, img.cols/downsample, CV_8UC1);
		auto t1 = std::chrono::high_resolution_clock::now();
		mg.createMask(img, mask);
		auto t2 = std::chrono::high_resolution_clock::now();
	    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
	    std::cout << "Mask created in " << fp_ms.count() << "ms" << std::endl;
		cv::imshow("mask", mask);
		cv::imshow("img", img);
		if (cv::waitKey(30) == 27) 
   		{
       	 	std::cout << "esc key is pressed by user" << std::endl;
        	break; 
   		}
	}

}


void video(std::string path)
{
	cv::VideoCapture cap(path);
	cv::VideoWriter vw;
	cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);


	cv::Mat img;
	cap >> img;
	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	//vw = cv::VideoWriter(std::string(argv[2])+"-output.avi", CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);

	cam_loop(cap);

}

void camera(std::string path)
{
	cv::VideoCapture cap(atoi(path.c_str()));
	cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

	cam_loop(cap);
}

int main(int argc, char** argv)
{
	std::string path = std::string(argv[2]);

	if (argv[1] == std::string("img"))
		image(path);
	else if(argv[1] == std::string("video"))
		video(path);
	else if(argv[1] == std::string("cam"))
		camera(path);
	else
	{
		std::cout << "ERROR : parameters are missing" << std::endl;
		std::cout << "Images : img directory_path" << std::endl << "Video : video video_path " << std::endl
		<< "Camera : cam #cam" << std::endl;
	}

	return 0;
}