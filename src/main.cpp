#include <iostream>
#include "utils.h"
#include "mgg_substractor.h"

const int nb_frame_init = 10;

void cam_loop(cv::VideoCapture& cap)
{
	std::vector<cv::Mat> data;
	for(int i = 0; i < nb_frame_init; i++)
	{
		cv::Mat img;
		cap >> img;
		data.push_back(img);
	}

	MGGBackgroundSubstractor mg;
	mg.init(data);

	while(1)
	{
		cv::Mat img;
		cap >> img;
		cv::imshow("MyWindow", img);
		if (cv::waitKey(30) == 27) 
   		{
       	 	std::cout << "esc key is pressed by user" << std::endl;
        	break; 
   		}
	}
}


void image(std::string path)
{
	std::vector<std::string> files = open_files(path);
	std::vector<cv::Mat> data;
	for(int i = 0; i < nb_frame_init; i++)
	{
		cv::Mat img = cv::imread(std::string(path) + files[i], CV_LOAD_IMAGE_COLOR);
		cv::cvtColor(img, img, CV_BGR2GRAY);
		data.push_back(img);
	}

	MGGBackgroundSubstractor mg;
	mg.init(data);
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