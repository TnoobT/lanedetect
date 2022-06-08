#include "lane.hpp"

using namespace std;
using namespace cv; 
using namespace MNN;

int main()
{
    string img_path = "../img/1.png";


    Mat src_img = imread(img_path);
    Mat image = src_img.clone();
    LaneDetect lanedetect;
    const float* heatmap = lanedetect.inference(image);
    auto lanes = lanedetect.decodeHeatmap(heatmap);
    lanedetect.showImg(src_img,lanes);

    return 0;




}