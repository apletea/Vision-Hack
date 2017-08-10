#include <iostream>
#include <opencv2/videoio.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <dirent.h>


std::string dir_path =  "/home/dzkamarouski/validationset/" ;
std::string model_txt = "/home/dzkamarouski/deep-learning-traffic-lights-master/model/deploy.prototxt";
std::string model_bin = "/home/dzkamarouski/deep-learning-traffic-lights-master/model/train_squeezenet_trainval_manual_p2__iter_3817.caffemodel";

cv::dnn::Net net = cv::dnn::readNetFromCaffe(model_txt, model_bin);

static void getMaxClass(const cv::Mat &probBlob, int *classId, double *classProb)
{
    cv::Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
    cv::Point classNumber;
    minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}



bool find_green(cv::Mat & mat)
{
    cv::Mat input_blob = cv::dnn::blobFromImage(mat,1,cv::Size(224,224), cv::Scalar(104,117,123));
    cv::Mat prob;
    cv::TickMeter t;
   // for (int i = 0; i < 10; ++i)
   // {
        CV_TRACE_REGION("forward");
        net.setInput(input_blob, "data");
        t.start();
        prob = net.forward("prob");
        t.stop();
   // }

    int classId;
    double classProb;
    getMaxClass(prob, &classId, &classProb);//find the best class
//    std::cout << "Best class: #" << classId << " '" << std::endl;
//    std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
//    std::cout << "Time: " << (double)t.getTimeMilli() / t.getCounter() << " ms (average from " << t.getCounter() << " iterations)" << std::endl;
    return classId == 2;
}


std::vector<std::string> split(const std::string &text, char sep)
{
    std::vector<std::string> tokens;
    std::size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != std::string::npos)
    {
        tokens.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(text.substr(start));
    return tokens;
}



int main()
{

    DIR *dirp;
    struct dirent *directory;
    dirp = opendir(dir_path.data());
    while ((directory = readdir(dirp)) != NULL)
    {
        if (directory->d_name == "." || directory->d_name == "..")
            continue;
        std::string  video_file_name  = dir_path + directory->d_name;
        //std::cout << video_file_name << std::endl;
      //  std::cout << directory->d_name << std::endl;
        cv::VideoCapture vc(video_file_name);
        if (!vc.isOpened()){
            vc.open("/home/dzkamarouski/video.mp4");
        }

        cv::Mat mat;
        int cap_num = 0;
        while(true)
        {
            vc >> mat;
            cv::waitKey(1);
            if (mat.empty())
                break;
            cv::imshow("l",mat);
            if (find_green(mat))
                break;
            cap_num++;
            //std::cout << "Frame number :" <<  cap_num << std::endl;
        }
        std::cout << directory->d_name << "  " << (mat.empty() ? -1 : cap_num) << std::endl;
        //std::cout << "================================================================================================" << std::endl;

    }
    closedir(dirp);



    return 0;
}

