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
std::string file = "ideal.txt";
std::string full_path = dir_path + file;
std::string save_red = "/home/dzkamarouski/red_light/";
std::string save_green = "/home/dzkamarouski/green_light/";

int main()
{
    std::ifstream ideal(full_path.data());
    std::string video_file_name;
    int frame_num;
    while ( ideal >> video_file_name >> frame_num )
    {
        frame_num = frame_num == -1 ? 500 : frame_num;
        std::string  path_to_video  = dir_path  + video_file_name ;
        cv::VideoCapture vc(path_to_video);
        if (!vc.isOpened())
            continue;
        cv::Mat mat;
        int cap_num = 0;
        while (true)
        {
            vc >> mat;
            cv::waitKey(1);
            if (mat.empty())
                break;
            std::string path_to_save = (cap_num >= frame_num ? save_green : save_red )+ video_file_name.data() + "_" + std::to_string(frame_num) + "_" + std::to_string(cap_num) + ".jpg";
            cv::imwrite(path_to_save, mat);
            cap_num++;
        }
    }
}
