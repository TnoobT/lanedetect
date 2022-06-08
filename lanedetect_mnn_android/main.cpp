#include "lane.hpp"

using namespace std;
using namespace cv; 
using namespace MNN;

int main()
{
    string img_path = "../img/1.png";
    const char* model_path = "../models/mlsd_with_max_sigmoid.mnn";

    Mat src_img = imread(img_path);
    Mat image = src_img.clone();
    LaneDetect lanedetect(model_path);
    const float* heatmap = lanedetect.inference(image);
    auto lanes = lanedetect.decodeHeatmap(heatmap);
    lanedetect.showImg(src_img,lanes);

    return 0;




}