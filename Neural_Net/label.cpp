#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <dirent.h>
#include <fstream>

std::string dir_path =  "/home/dzkamarouski/trainset/" ;
std::string file = "labels1.txt";
std::string full_path = dir_path + file;
std::string save_red = "/home/dzkamarouski/red_light/";
std::string save_green = "/home/dzkamarouski/green_light/";

int main()
{
    std::ofstream labels(file);
    DIR *dirp;
    struct dirent * directory;
    dirp = opendir(save_green.data());
    while( (directory = readdir(dirp)) != NULL)
    {
        labels << save_red.data() << directory->d_name << " " << std::to_string(1) << "\n";
    }

}
