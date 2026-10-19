#include "detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>


// --------------------------------------------------
// Constructor
// --------------------------------------------------

Detector::Detector(const std::string& model_path)
    : env_(
        ORT_LOGGING_LEVEL_WARNING,
        "visdrone"
      ),
      session_options_{},
      session_(
        env_,
        model_path.c_str(),
        session_options_
      ),
      memory_info_(
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        )
      )
{
    Ort::AllocatorWithDefaultOptions allocator;

    // Names: copy into std::string so they outlive the allocated pointers
    auto input_name =
        session_.GetInputNameAllocated(0, allocator);

    auto output_name =
        session_.GetOutputNameAllocated(0, allocator);

    input_name_ = input_name.get();
    output_name_ = output_name.get();

    // Input shape: [1, 3, H, W]
    auto input_shape =
        session_
            .GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    // Output shape: [1, 4 + classes, candidates]
    auto output_shape =
        session_
            .GetOutputTypeInfo(0)
            .GetTensorTypeAndShapeInfo()
            .GetShape();

    if (input_shape.size() != 4)
    {
        throw std::runtime_error("Unexpected model input shape");
    }

    if (output_shape.size() != 3)
    {
        throw std::runtime_error("Unexpected model output shape");
    }

    if (input_shape[2] != input_shape[3])
    {
        throw std::runtime_error("Expected a square model input");
    }

    input_size_ = static_cast<int>(input_shape[2]);
    num_candidates_ = static_cast<int>(output_shape[2]);
    num_classes_ = static_cast<int>(output_shape[1] - 4);

    if (input_size_ <= 0)
    {
        throw std::runtime_error("Invalid model input size");
    }

    if (num_candidates_ <= 0)
    {
        throw std::runtime_error("Invalid candidate count");
    }

    if (num_classes_ <= 0)
    {
        throw std::runtime_error("Invalid class count");
    }

    // Allocate the input tensor buffer ONCE.
    // preprocess() only overwrites its contents.
    input_buffer_.resize(
        static_cast<size_t>(3) * input_size_ * input_size_
    );

    input_shape_ = {
        1,
        3,
        input_size_,
        input_size_
    };
}


// --------------------------------------------------
// Preprocess
//   letterbox -> BGR->RGB -> /255 -> HWC->CHW
//   The last three steps are done in ONE pass over the
//   padded uint8 image, written straight into input_buffer_.
// --------------------------------------------------

Detector::PreprocessResult Detector::preprocess(
    const cv::Mat& image
)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty");
    }

    const int W = input_size_;

    const int orig_w = image.cols;
    const int orig_h = image.rows;

    // Preserve aspect ratio
    const float scale =
        std::min(
            static_cast<float>(W) / orig_w,
            static_cast<float>(W) / orig_h
        );

    const int new_w =
        static_cast<int>(std::round(orig_w * scale));

    const int new_h =
        static_cast<int>(std::round(orig_h * scale));

    // Resize into a reused member Mat (no new allocation after the first call)
    cv::resize(
        image,
        resized_,
        cv::Size(new_w, new_h),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    // Letterbox padding
    const int pad_x = (W - new_w) / 2;
    const int pad_y = (W - new_h) / 2;

    const int pad_right = W - new_w - pad_x;
    const int pad_bottom = W - new_h - pad_y;

    cv::copyMakeBorder(
        resized_,
        padded_,
        pad_y,
        pad_bottom,
        pad_x,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // One pass: uint8 BGR (HWC) -> float RGB (CHW), scaled to 0..1
    const int plane = W * W;

    float* r_plane = input_buffer_.data();
    float* g_plane = r_plane + plane;
    float* b_plane = g_plane + plane;

    // Same scale factor that convertTo(CV_32F, 1.0 / 255.0) used
    const float k = 1.0f / 255.0f;

    for (int y = 0; y < W; ++y)
    {
        const uchar* row = padded_.ptr<uchar>(y);
        const int offset = y * W;

        for (int x = 0; x < W; ++x)
        {
            const uchar* px = row + 3 * x;   // OpenCV order: B, G, R

            b_plane[offset + x] = px[0] * k;
            g_plane[offset + x] = px[1] * k;
            r_plane[offset + x] = px[2] * k;
        }
    }

    return {
        scale,
        pad_x,
        pad_y
    };
}


// --------------------------------------------------
// Detect
// --------------------------------------------------

std::vector<Detection> Detector::detect(
    const cv::Mat& image,
    float conf_threshold,
    float iou_threshold
)
{
    if (image.empty())
    {
        throw std::runtime_error("Input image is empty");
    }

    // ---------------- Preprocess ----------------
    auto pre_start = std::chrono::steady_clock::now();

    const PreprocessResult prep = preprocess(image);

    auto pre_end = std::chrono::steady_clock::now();

    // Wrap the reused buffer (no copy, just a view)
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info_,
            input_buffer_.data(),
            input_buffer_.size(),
            input_shape_.data(),
            input_shape_.size()
        );

    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    // ---------------- Inference ----------------
    auto infer_start = std::chrono::steady_clock::now();

    std::vector<Ort::Value> outputs =
        session_.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    auto infer_end = std::chrono::steady_clock::now();

    if (outputs.empty())
    {
        throw std::runtime_error("ONNX Runtime returned no outputs");
    }

    const float* data = outputs[0].GetTensorData<float>();

    // ---------------- Postprocess ----------------
    auto post_start = std::chrono::steady_clock::now();

    std::vector<Detection> detections =
        postprocess(
            data,
            prep,
            image.cols,
            image.rows,
            conf_threshold,
            iou_threshold
        );

    auto post_end = std::chrono::steady_clock::now();

    // ---------------- Store timings ----------------
    last_timings_.pre_ms =
        std::chrono::duration<double, std::milli>(pre_end - pre_start).count();

    last_timings_.infer_ms =
        std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

    last_timings_.post_ms =
        std::chrono::duration<double, std::milli>(post_end - post_start).count();

    return detections;
}


// --------------------------------------------------
// Postprocess (unchanged logic)
// --------------------------------------------------

std::vector<Detection> Detector::postprocess(
    const float* data,
    const PreprocessResult& prep,
    int orig_w,
    int orig_h,
    float conf_threshold,
    float iou_threshold
)
{
    std::vector<Detection> candidates;
    candidates.reserve(static_cast<size_t>(num_candidates_));

    // Decode every candidate. Value j of candidate i is at data[j * N + i]
    for (int i = 0; i < num_candidates_; ++i)
    {
        float best_score = -1.0f;
        int best_class = -1;

        for (int c = 0; c < num_classes_; ++c)
        {
            const float score = data[(4 + c) * num_candidates_ + i];

            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold)
        {
            continue;
        }

        const float cx = data[0 * num_candidates_ + i];
        const float cy = data[1 * num_candidates_ + i];
        const float w  = data[2 * num_candidates_ + i];
        const float h  = data[3 * num_candidates_ + i];

        Detection det;

        // cxcywh -> xyxy (letterbox coordinates)
        det.x1 = cx - w / 2.0f;
        det.y1 = cy - h / 2.0f;
        det.x2 = cx + w / 2.0f;
        det.y2 = cy + h / 2.0f;

        det.class_id = best_class;
        det.score = best_score;

        candidates.push_back(det);
    }

    // Class-aware NMS in letterbox coordinates
    std::vector<Detection> final_detections =
        nms(candidates, iou_threshold);

    // Undo letterbox and clip to the original image
    const float max_x = static_cast<float>(orig_w);
    const float max_y = static_cast<float>(orig_h);

    for (Detection& det : final_detections)
    {
        det.x1 = (det.x1 - prep.pad_x) / prep.scale;
        det.y1 = (det.y1 - prep.pad_y) / prep.scale;
        det.x2 = (det.x2 - prep.pad_x) / prep.scale;
        det.y2 = (det.y2 - prep.pad_y) / prep.scale;

        det.x1 = std::clamp(det.x1, 0.0f, max_x);
        det.y1 = std::clamp(det.y1, 0.0f, max_y);
        det.x2 = std::clamp(det.x2, 0.0f, max_x);
        det.y2 = std::clamp(det.y2, 0.0f, max_y);
    }

    // Highest confidence first
    std::sort(
        final_detections.begin(),
        final_detections.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.score > b.score;
        }
    );

    return final_detections;
}


// --------------------------------------------------
// IoU
// --------------------------------------------------

float Detector::iou(
    const Detection& a,
    const Detection& b
)
{
    const float inter_left   = std::max(a.x1, b.x1);
    const float inter_top    = std::max(a.y1, b.y1);
    const float inter_right  = std::min(a.x2, b.x2);
    const float inter_bottom = std::min(a.y2, b.y2);

    const float inter_width  = std::max(0.0f, inter_right - inter_left);
    const float inter_height = std::max(0.0f, inter_bottom - inter_top);

    const float intersection = inter_width * inter_height;

    const float area_a =
        std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);

    const float area_b =
        std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);

    const float union_area = area_a + area_b - intersection;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return intersection / union_area;
}


// --------------------------------------------------
// Class-aware NMS
// --------------------------------------------------

std::vector<Detection> Detector::nms(
    std::vector<Detection> detections,
    float iou_threshold
)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.score > b.score;
        }
    );

    std::vector<Detection> kept;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
        {
            continue;
        }

        kept.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppressed[j])
            {
                continue;
            }

            // Only suppress detections of the same class
            if (detections[i].class_id != detections[j].class_id)
            {
                continue;
            }

            // Match Python: suppress only if IoU > threshold
            if (iou(detections[i], detections[j]) > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}