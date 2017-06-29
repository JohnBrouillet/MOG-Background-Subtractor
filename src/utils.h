#ifndef UTILS_H
#define UTILS_H

#include <dirent.h>
#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <sys/stat.h>


std::vector<std::string> open_files(std::string path = ".");
bool checkMaskDirExist(std::string directory);

#endif