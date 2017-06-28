#include <iostream>
#include "utils.h"

int main(int argc, char** argv)
{
	if (argv[1] == std::string("img"))
	{
		std::vector<std::string> files = open_files(argv[2]);
	
		//cv::Mat img = cv::imread(std::string(argv[2]) + files[0], CV_LOAD_IMAGE_COLOR);
	}
	else if(argv[1] == std::string("video"))
	{
		cv::VideoCapture cap(0);
		cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
		if(argv[2] == std::string("cam"))
			cap = cv::VideoCapture(0);
		else
			cap = cv::VideoCapture(argv[2]);

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
	else
	{
		std::cout << "ERROR : parameters are missing" << std::endl;
		std::cout << "Images : img directory_path" << std::endl << "Video : video video_path (or cam for camera)" << std::endl;
	}

	return 0;
}