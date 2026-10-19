#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cstdint>
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


struct Timings
{
    double pre_ms = 0.0;
    double infer_ms = 0.0;
    double post_ms = 0.0;
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

    const Timings& last_timings() const
    {
        return last_timings_;
    }

    int input_size() const
    {
        return input_size_;
    }

private:
    // Only the letterbox parameters. The tensor itself lives in
    // input_buffer_ and is reused on every call.
    struct PreprocessResult
    {
        float scale;
        int pad_x;
        int pad_y;
    };

    PreprocessResult preprocess(const cv::Mat& image);

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

    // --------------------------------------------------
    // ONNX Runtime objects
    // ORDER MATTERS: members are constructed in the order they
    // are declared here, and session_ needs env_ and
    // session_options_ to exist first.
    // --------------------------------------------------
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    std::string input_name_;
    std::string output_name_;

    int input_size_ = 0;
    int num_candidates_ = 0;
    int num_classes_ = 0;

    // --------------------------------------------------
    // Reused buffers (allocated once, overwritten every call)
    // --------------------------------------------------
    std::vector<float> input_buffer_;          // 3 * H * W, CHW, RGB, 0..1
    std::array<int64_t, 4> input_shape_{};     // {1, 3, H, W}

    cv::Mat resized_;   // cv::Mat reuses its memory when size/type are unchanged
    cv::Mat padded_;

    Timings last_timings_;
};