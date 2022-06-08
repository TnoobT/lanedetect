#include "lane.hpp"

using namespace std;
using namespace cv; 
using namespace MNN;

int main()
{
    string img_path = "../img/1.png";
    string mnn_path = "../models/mlsd_with_max_sigmoid.mnn";
    Mat src_img = imread(img_path);
    Mat image = src_img.clone();
    LaneDetect lanedetect(mnn_path,true);
    std::vector<LaneDetect::Lanes> lanes = lanedetect.detect(image,NULL,image.cols,image.rows,0.2,20);
    

    return 0;




}