#include "lane.hpp"
// #include "omp.h"

bool LaneDetect::hasGPU = false;
bool LaneDetect::toUseGPU = false;
LaneDetect *LaneDetect::detector = nullptr;

LaneDetect::LaneDetect(const std::string &mnn_path, bool useGPU)
{
   
    toUseGPU = hasGPU && useGPU;
    m_net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path.c_str()));
    
    m_config.numThread = 4;
    if (useGPU) {
        m_config.type = MNN_FORWARD_OPENCL;
    }
    m_config.backupType = MNN_FORWARD_CPU;
    m_backend_config.memory = MNN::BackendConfig::Memory_Normal;  // 内存
    m_backend_config.power = MNN::BackendConfig::Power_Normal;  // 功耗
    m_backend_config.precision = MNN::BackendConfig::PrecisionMode::Precision_Low;  // 精度
    m_config.backendConfig = &m_backend_config;
    m_session = m_net->createSession(m_config); //创建session
    m_inTensor = m_net->getSessionInput(m_session, NULL);
}

LaneDetect::~LaneDetect()
{
    m_net->releaseModel();
    m_net->releaseSession(m_session);
    m_lanes.clear();
}

inline int LaneDetect::clip(float value)
{
    if (value > 0 && value < m_input_size)
        return int(value);
    else if (value < 0)
        return 1;
    else
        return m_input_size - 1;

}

void LaneDetect::showImg(const cv::Mat& img,std::vector<LaneDetect::Lanes> Lanes)
{
    cv::Mat tmp_img = img.clone();
    int h = tmp_img.rows;
    int w = tmp_img.cols;
    
    for(auto line:Lanes)
    {
        float x1 = line.x1;
        float y1 = line.y1;
        float x2 = line.x2;
        float y2 = line.y2;
        x1 = clip(w * x1 / (m_input_size / 2));
        x2 = clip(w * x2 / (m_input_size / 2));
        y1 = clip(h * y1 / (m_input_size / 2));
        y2 = clip(h * y2 / (m_input_size / 2));

        cv::line(tmp_img,cv::Point(x1,y1),cv::Point(x2,y2), cv::Scalar(0, 0, 255));
    }

    cv::imshow("img",tmp_img);
    cv::waitKey(1);
    return ;
}



std::vector<LaneDetect::Lanes> LaneDetect::decodeHeatmap(const float* hm)
{   
    // 线段中心点(256*256),线段偏移(4*256*256)
    const float*  displacement = hm+m_hm_size*m_hm_size;
    // exp(center,center);
    std::vector<float> center;
    for (int i = 0;i < m_hm_size*m_hm_size; i++)
    {
        center.push_back( hm[i] ); // mlsd.mnn原始需要1/(exp(-hm[i]) + 1)
    }
    center.resize(m_hm_size*m_hm_size);

    std::vector<int> index(center.size(), 0);
    for (int i = 0 ; i != index.size() ; i++) {
        index[i] = i;
    }
    sort(index.begin(), index.end(),
        [&](const int& a, const int& b) {
            return (center[a] > center[b]); // 从大到小排序
        }
    );
    
    for (int i = 0; i < index.size(); i++)
    {
        int yy = index[i]/m_hm_size; // 除以宽得行号
        int xx = index[i]%m_hm_size; // 取余宽得列号
        Lanes Lane;
        Lane.x1 = xx + displacement[index[i] + 0*m_hm_size*m_hm_size];
        Lane.y1 = yy + displacement[index[i] + 1*m_hm_size*m_hm_size];
        Lane.x2 = xx + displacement[index[i] + 2*m_hm_size*m_hm_size];
        Lane.y2 = yy + displacement[index[i] + 3*m_hm_size*m_hm_size];
        Lane.lens = sqrt(pow(Lane.x1 - Lane.x2,2) + pow(Lane.y1 - Lane.y2,2));
        Lane.conf = center[index[i]];

        if (center[index[i]] > m_score_thresh && m_lanes.size() < m_top_k)
        {
            if ( Lane.lens > m_min_len)
                m_lanes.push_back(Lane);
        }
        else
            break;
    }
    
    return m_lanes;

}


void LaneDetect::processImg(const cv::Mat& img)
{   
    cv::Mat preImage = img.clone();
    cv::cvtColor(preImage,preImage,cv::COLOR_BGR2RGB);
    cv::resize(preImage,preImage,cv::Size(m_input_size,m_input_size));
	preImage.convertTo(preImage,CV_32FC3,1/127.5,-1); // (img / 127.5) - 1.0
    std::vector<cv::Mat> bgrChannels(3);
    cv::split(preImage, bgrChannels);
    std::vector<float> chwImage;
    for (auto i = 0; i < bgrChannels.size(); i++)
    {  
        //HWC->CHW
        std::vector<float> data = std::vector<float>(bgrChannels[i].reshape(1, preImage.cols * preImage.rows));
        chwImage.insert(chwImage.end(), data.begin(), data.end());
    }
    
	auto nchw_Tensor = new MNN::Tensor(m_inTensor, MNN::Tensor::CAFFE); // tensorflow: nhwc, caffe: nchw, caffe_c4: nc4hw4
    ::memcpy(nchw_Tensor->host<float>(), chwImage.data(), nchw_Tensor->elementSize() * 4);
    m_inTensor->copyFromHostTensor(nchw_Tensor);
    // delete nchw_Tensor;
}

std::vector<LaneDetect::Lanes> LaneDetect::inference(const cv::Mat& img)
{
    processImg(img);
    m_net->runSession(m_session);
    auto output= m_net->getSessionOutput(m_session, NULL);
    auto nchwTensor = new MNN::Tensor(output, MNN::Tensor::CAFFE);
    output->copyToHostTensor(nchwTensor);
    float* score = nchwTensor->host<float>(); // 得到结果指针
    std::vector<Lanes> lanes = decodeHeatmap(score);
    showImg(img,lanes);
    return lanes;
}