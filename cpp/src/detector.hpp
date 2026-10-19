#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;

    int class_id;
    float score;
};

class Detector
{
public:
    explicit Detector(const std::string& model_path);

    std::vector<Detection> detect(
        const cv::Mat& image,
        float conf_threshold = 0.25f,
        float iou_threshold = 0.7f
    );

private:
    struct PreprocessResult
    {
        std::vector<float> tensor;
        float scale;
        int pad_x;
        int pad_y;
    };

    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;

    std::string input_name_;
    std::string output_name_;

    int input_size_;
    int num_candidates_;
    int num_classes_;

    PreprocessResult preprocess(
        const cv::Mat& image
    );

    std::vector<Detection> postprocess(
        const float* data,
        const PreprocessResult& prep,
        int orig_w,
        int orig_h,
        float conf_threshold,
        float iou_threshold
    );

    static float iou(
        const Detection& a,
        const Detection& b
    );

    static std::vector<Detection> nms(
        std::vector<Detection> detections,
        float iou_threshold
    );
};