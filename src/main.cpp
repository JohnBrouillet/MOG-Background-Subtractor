#include <iostream>
#include "utils.h"
#include "mog_subtractor.h"

const int nb_frame_init = 50;
const int nb_gauss = 5;
const int downsample = 1;
const int max_area = 10000;
int area_thresh = 500;

void createBox(cv::Mat& mask, cv::Mat& img)
{
	std::vector<std::vector<cv::Point> > contours; 
    std::vector<cv::Vec4i> hierarchy;
    Rect bounding_rect;
	cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_TC89_KCOS );

	for( int i = 0; i< contours.size(); i++ )
    {   
        double area = cv::contourArea(contours[i], false);
        if(area > area_thresh){
            bounding_rect = cv::boundingRect(contours[i]);
            cv::rectangle(img, bounding_rect, Scalar(0,255,0), 1, 8, 0);
        }       
    }
}

void cam_loop(cv::VideoCapture& cap, cv::VideoWriter vw, bool save = false)
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
		
		cv::Mat img, img_clone;
		cap >> img;

		img_clone = img.clone();
		cv::resize(img_clone, img_clone, cv::Size(), 1.0/downsample, 1.0/downsample);

		Mat mask = mg.createMask(img);
		createBox(mask, img_clone);
		cv::imshow("Camera", img_clone);
		cv::imshow("Mask", mask);
		cv::createTrackbar("Threshold", "Camera", &area_thresh, max_area);
		if (cv::waitKey(30) == 27) 
   		{
       	 	std::cout << "esc key is pressed by user" << std::endl;
        	break; 
   		}
		
   		if(save)
   			vw.write(img_clone);

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
	checkMaskDirExist(path);
	std::vector<std::string> files = open_files(path);
	std::vector<cv::Mat> data;
	for(int i = 0; i < nb_frame_init; i++)
	{
		cv::Mat img = cv::imread(std::string(path) + files[i], CV_LOAD_IMAGE_COLOR);
		data.push_back(img);
	}

	MOGBackgroundSubtraction mg(nb_gauss, downsample);
	mg.init(data);

	auto t1 = std::chrono::high_resolution_clock::now();
	for(int i = nb_frame_init; i < files.size()-1; i++)
	{
		cv::Mat img = cv::imread(std::string(path) + files[i], CV_LOAD_IMAGE_COLOR);
		cv::Mat img_clone = img.clone();

		
		Mat mask = mg.createMask(img);
	
	    createBox(mask, img_clone);
		cv::imshow("mask", mask);
		cv::imshow("img", img_clone);
		imwrite(path + "/mask/" + files[i] + ".bmp", mask);
		if (cv::waitKey(30) == 27) 
   		{
       	 	std::cout << "esc key is pressed by user" << std::endl;
        	break; 
   		}
	}
	auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> fp_ms = t2 - t1;
    std::cout << files.size() - nb_frame_init << " masks computed in "<< fp_ms.count()*1e-3 << "s" << std::endl;

}


void video(std::string path)
{
	cv::VideoCapture cap(path);
	cv::VideoWriter vw;

	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	vw = cv::VideoWriter(path + "-output.avi", CV_FOURCC('M','J','P','G'),25, Size(frame_width,frame_height),true);

	cam_loop(cap, vw, true);

}

void camera(std::string path)
{
	cv::VideoCapture cap(atoi(path.c_str()));
	cv::VideoWriter vw;

	int frame_width = cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
	vw = cv::VideoWriter("cam-output.avi", CV_FOURCC('M','J','P','G'),25, Size(frame_width,frame_height),true);

	cam_loop(cap, vw, true);
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