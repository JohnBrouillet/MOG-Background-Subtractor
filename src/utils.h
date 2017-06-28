#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>


#include "opencv_include.h"

std::vector<std::string> open_files(std::string path = ".");
bool save(cv::Mat& img, std::string directory, std::string name, std::string format);

#endif