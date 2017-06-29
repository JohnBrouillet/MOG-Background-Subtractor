#include <iostream>
#include "utils.h"
#include "mgg_substractor.h"

const int nb_frame_init = 50;

int main(int argc, char** argv)
{
	if (argv[1] == std::string("img"))
	{
				cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

		std::vector<std::string> files = open_files(argv[2]);
		std::vector<cv::Mat> data;
		for(int i = 0; i < nb_frame_init; i++)
		{
 			cv::Mat img = cv::imread(std::string(argv[2]) + files[i], CV_LOAD_IMAGE_COLOR);
			cv::cvtColor(img, img, CV_BGR2GRAY);
			cv::imshow("MyWindow", img);
			data.push_back(img);
		}

		MGGBackgroundSubstractor mg(3,1);
		mg.init(data);
	}
	else if(argv[1] == std::string("video"))
	{
		cv::VideoCapture cap(0);
		cv::VideoWriter vw;
		cv::namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);

		if(argv[2] == std::string("cam"))
			cap = cv::VideoCapture(0);
		else
		{	
			cap = cv::VideoCapture(argv[2]);
			cv::Mat img;
			cap >> img;
			int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
   			int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
			//vw = cv::VideoWriter(argv[2]+"-output.avi", CV_FOURCC('M','J','P','G'),10, Size(frame_width,frame_height),true);
		}

		std::vector<cv::Mat> data;
		for(int i = 0; i < nb_frame_init; i++)
		{
			cv::Mat img;
			cap >> img;
			data.push_back(img);
		}

		MGGBackgroundSubstractor mg(3,1);
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
	else
	{
		std::cout << "ERROR : parameters are missing" << std::endl;
		std::cout << "Images : img directory_path" << std::endl << "Video : video video_path (or cam for camera)" << std::endl;
	}

	return 0;
}