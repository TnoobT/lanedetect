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
        // 数据预处理，放入tensor中
        void processImg(const cv::Mat& img);
        // 推理，输出结果指针
        std::vector<Lanes> inference(const cv::Mat& img);
        std::vector<Lanes> decodeHeatmap(const float* heatmap);
        void showImg(const cv::Mat& img,std::vector<Lanes> Lanes);
        inline int clip(float value);
    private:
        std::shared_ptr<MNN::Interpreter> m_net = nullptr;
        MNN::ScheduleConfig m_config;
        MNN::Session *m_session = nullptr;
        MNN::Tensor *m_inTensor = nullptr;
        MNN::BackendConfig m_backend_config;

        std::vector<Lanes> m_lanes;
        float m_score_thresh = 0.2; // 阈值
        int m_input_size     = 512; // 输入尺寸
        int m_hm_size  = 256; // 特征图大小
        int m_min_len  = 20; // 预测线段的最短长度
        int m_top_k    = 200; // 取200条线
       
    
    public: // for android
        static LaneDetect *detector;
        static bool hasGPU;
        static bool toUseGPU;
};


#endif