#ifndef __LANE_H__
#define __LANE_H__

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "MNN/Interpreter.hpp"
#include "MNN/ImageProcess.hpp"

class LaneDetect
{
    public:
        typedef struct Lanes
        {
            float x1;  // 起点
            float y1;
            float x2;  // 终点
            float y2;
            float lens;
            float conf;

        }Lanes;
    

    public:
        LaneDetect(const std::string &mnn_path, bool useGPU);
        ~LaneDetect();   
        std::vector<Lanes> detect(const cv::Mat& img,unsigned char* image_bytes, int width, int height, double threshold, double lens_threshold);
        std::vector<Lanes> decodeHeatmap(const float* heatmap, int width, int height, double threshold, double lens_threshold);
        void showImg(const cv::Mat& img,std::vector<Lanes> Lanes);
        inline int clip(float value);
    private:
        std::shared_ptr<MNN::Interpreter> m_net = nullptr;
        MNN::ScheduleConfig m_config;
        MNN::Session *m_session = nullptr;
        MNN::Tensor *m_inTensor = nullptr;
        std::shared_ptr<MNN::CV::ImageProcess> pretreat = nullptr;
        MNN::BackendConfig m_backend_config;

        const float m_mean_vals[3] = { 127.5f, 127.5f, 127.5f };
        const float m_norm_vals[3] = { 1/127.5f, 1/127.5f, 1/127.5f };
       
        int m_input_size     = 512; // 输入尺寸
        int m_hm_size  = 256; // 特征图大小
        int m_top_k    = 200; // 取200条线

    public:
        static LaneDetect *detector;
        static bool hasGPU;
        static bool toUseGPU;

};


#endif