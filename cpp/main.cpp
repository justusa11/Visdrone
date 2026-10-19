#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <array>
#include <chrono>
#include <limits>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct PreprocessResult
{
    std::vector<float> tensor;
    float scale;
    int pad_x;
    int pad_y;
};

PreprocessResult preprocess(
    const cv::Mat& image,
    int input_size = 640
)
{
    const int orig_w = image.cols;
    const int orig_h= image.rows;

    const float scale = std::min(
        static_cast<float>(input_size) / orig_w,
        static_cast<float>(input_size) / orig_h
    );

    const int new_w = static_cast<int>(
        std::round(orig_w * scale)
    );

    const int new_h = static_cast<int>(
        std::round(orig_h * scale)
    );

    cv::Mat resized;

    cv::resize(
        image,
        resized,
        cv::Size(new_w, new_h),
        0.0,
        0.0,
        cv::INTER_LINEAR
    );

    const int pad_x = (input_size - new_w) / 2;
    const int pad_y = (input_size - new_h) / 2;

    const int pad_right =
        input_size - new_w - pad_x;

    const int pad_bottom =
        input_size - new_h - pad_y;

    cv::Mat padded;

    cv::copyMakeBorder(
        resized,
        padded,
        pad_y,
        pad_bottom,
        pad_x,
        pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    cv::Mat rgb;

    cv::cvtColor(
        padded,
        rgb,
        cv::COLOR_BGR2RGB
    );

    cv::Mat float_img;

    rgb.convertTo(
        float_img,
        CV_32F,
        1.0 / 255.0
    );

    std::vector<float> tensor(
        3 * input_size * input_size
    );

    for (int c = 0; c < 3; ++c)
    {
        for (int y = 0; y < input_size; ++y)
        {
            for (int x = 0; x < input_size; ++x)
            {
                const cv::Vec3f pixel =
                    float_img.at<cv::Vec3f>(y, x);

                const int index =
                    c * input_size * input_size
                    + y * input_size
                    + x;

                tensor[index] = pixel[c];
            }
        }
    }

    return {
        std::move(tensor),
        scale,
        pad_x,
        pad_y
    };
}


void printShape(const std::vector<int64_t>& shape)
{
    std::cout << "[";

    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];

        if (i + 1 < shape.size())
        {
            std::cout << ", ";
        }
    }

    std::cout << "]";
}

// --------------------------------------------------
// Detection Struct
// --------------------------------------------------
struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;

    int class_id;
    float score;
};

// --------------------------------------------------
// IoU
// --------------------------------------------------
float iou(
    const Detection& a,
    const Detection& b
)
{
    const float inter_left = std::max(a.x1, b.x1);
    const float inter_top = std::max(a.y1, b.y1);
    const float inter_right = std::min(a.x2, b.x2);
    const float inter_bottom = std::min(a.y2, b.y2);

    const float inter_w =
        std::max(0.0f, inter_right - inter_left);

    const float inter_h =
        std::max(0.0f, inter_bottom - inter_top);

    const float intersection =
        inter_w * inter_h;

    const float area_a =
        std::max(0.0f, a.x2 - a.x1)
        * std::max(0.0f, a.y2 - a.y1);

    const float area_b =
        std::max(0.0f, b.x2 - b.x1)
        * std::max(0.0f, b.y2 - b.y1);

    const float union_area =
        area_a + area_b - intersection;

    if (union_area <= 0.0f)
    {
        return 0.0f;
    }

    return intersection / union_area;
}

// --------------------------------------------------
// Class-aware NMS
// --------------------------------------------------
std::vector<Detection> nms(
    std::vector<Detection> detections,
    float iou_threshold = 0.7f
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
    std::vector<bool> suppressed(
        detections.size(),
        false
    );

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

            // Class-aware NMS
            if (detections[i].class_id != detections[j].class_id)
            {
                continue;
            }

            if (iou(detections[i], detections[j]) > iou_threshold)
            {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <image_path> <model_path>\n";

        return 1;
    }

    const std::string imagePath = argv[1];
    const std::string modelPath = argv[2];

    // --------------------------------------------------
    // OpenCV sanity check
    // --------------------------------------------------

    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
    {
        std::cerr
            << "Failed to load image: "
            << imagePath
            << "\n";

        return 1;
    }

    PreprocessResult prep = preprocess(
    image,
    640
    );

    std::cout << "\nPreprocessing\n";
    std::cout << "Scale: " << prep.scale << "\n";
    std::cout << "Pad x: " << prep.pad_x << "\n";
    std::cout << "Pad y: " << prep.pad_y << "\n";
    std::cout << "Tensor size: " << prep.tensor.size() << "\n";

    const int plane = 640 * 640;

    const int row = 140;
    const int col = 0;

    std::cout
        << "Pixel R: "
        << prep.tensor[0 * plane + row * 640 + col]
        << "\n";

    std::cout
        << "Pixel G: "
        << prep.tensor[1 * plane + row * 640 + col]
        << "\n";

    std::cout
        << "Pixel B: "
        << prep.tensor[2 * plane + row * 640 + col]
        << "\n";

    std::cout << "First R: " << prep.tensor[0] << "\n";
    std::cout << "First G: " << prep.tensor[plane] << "\n";
    std::cout << "First B: " << prep.tensor[2 * plane] << "\n";

    double checksum = 0.0;

    for (float value : prep.tensor)
    {
        checksum += value;
    }

    std::cout 
        << std::fixed
        << std::setprecision(4)
        << "Tensor checksum: "  
        << checksum 
        << "\n";

    std::cout
        << "Image: "
        << image.cols
        << " x "
        << image.rows
        << ", channels = "
        << image.channels()
        << "\n\n";

    // --------------------------------------------------
    // ONNX Runtime environment
    // --------------------------------------------------

    Ort::Env env(
        ORT_LOGGING_LEVEL_WARNING,
        "visdrone"
    );

    Ort::SessionOptions sessionOptions;

    // CPU for now
    sessionOptions.SetIntraOpNumThreads(1);

    // --------------------------------------------------
    // Create model session
    // --------------------------------------------------

    #ifdef _WIN32
        std::wstring modelPathW(
            modelPath.begin(),
            modelPath.end()
        );

        Ort::Session session(
            env,
            modelPathW.c_str(),
            sessionOptions
        );
    #else
        Ort::Session session(
            env,
            modelPath.c_str(),
            sessionOptions
        );
    #endif

        Ort::AllocatorWithDefaultOptions allocator;

    // --------------------------------------------------
    // RUN ONNX MODEL
    // --------------------------------------------------

    // Input tensor shape: [1, 3, 640, 640]
    std::array<int64_t, 4> input_shape = {
        1,
        3,
        640,
        640
    };

    // Tell ONNX Runtime that our tensor lives in CPU memory
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator,
            OrtMemTypeDefault
        );

    // Create ONNX Runtime tensor.
    //
    // IMPORTANT:
    // This does NOT copy prep.tensor.
    // ONNX Runtime points directly at prep.tensor.data().
    // Therefore prep.tensor must remain alive until session.Run() finishes.
    Ort::Value input_tensor =
        Ort::Value::CreateTensor<float>(
            memory_info,
            prep.tensor.data(),
            prep.tensor.size(),
            input_shape.data(),
            input_shape.size()
        );

    // We already verified these names in Step 5.2
    const char* input_names[] = {
        "images"
    };

    const char* output_names[] = {
        "output0"
    };


    // --------------------------------------------------
    // RUN INFERENCE + TIME IT
    // --------------------------------------------------

    auto start = std::chrono::steady_clock::now();

    std::vector<Ort::Value> outputs =
        session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

    auto end = std::chrono::steady_clock::now();

    double inference_ms =
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    std::cout
        << "\nInference time: "
        << inference_ms
        << " ms\n";


    // --------------------------------------------------
    // CHECK OUTPUT SHAPE
    // --------------------------------------------------

    auto output_info =
        outputs[0].GetTensorTypeAndShapeInfo();

    std::vector<int64_t> output_shape =
        output_info.GetShape();

    std::cout << "Raw output shape: [";

    for (size_t i = 0; i < output_shape.size(); ++i)
    {
        std::cout << output_shape[i];

        if (i + 1 < output_shape.size())
        {
            std::cout << ", ";
        }
    }

    std::cout << "]\n";


    // --------------------------------------------------
    // READ RAW OUTPUT
    // --------------------------------------------------

    const float* data =
        outputs[0].GetTensorData<float>();

    // --------------------------------------------------
    // DECODE 8400 CANDIDATES
    // --------------------------------------------------
    constexpr int NUM_CANDIDATES = 8400;
    constexpr int NUM_CLASSES = 10;

    std::vector<Detection> candidates;

    for (int i = 0; i < NUM_CANDIDATES; ++i)
    {
        float best_score = -1.0f;
        int best_class = -1;

        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            const float score =
                data[(4 + c) * NUM_CANDIDATES + i];

            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < 0.25f)
        {
            continue;
        }

        const float cx =
            data[0 * NUM_CANDIDATES + i];

        const float cy =
            data[1 * NUM_CANDIDATES + i];

        const float w =
            data[2 * NUM_CANDIDATES + i];

        const float h =
            data[3 * NUM_CANDIDATES + i];

        Detection det;

        det.x1 = cx - w / 2.0f;
        det.y1 = cy - h / 2.0f;
        det.x2 = cx + w / 2.0f;
        det.y2 = cy + h / 2.0f;

        det.class_id = best_class;
        det.score = best_score;

        candidates.push_back(det);
    }

    // --------------------------------------------------
    // APPLY CLASS-AWARE NMS
    // --------------------------------------------------
    std::cout
        << "Candidates after score filter: "
        << candidates.size()
        << "\n";

    // --------------------------------------------------
    // APPLY CLASS-AWARE NMS
    // --------------------------------------------------
    std::vector<Detection> final_detections =
        nms (
            candidates,
            0.7f
        );

    std::cout
        << "After NMS: "
        << final_detections.size()
        << "\n";

    // --------------------------------------------------
    // MAP BOXES BACK TO ORIGINAL
    // --------------------------------------------------
    for (Detection& det : final_detections)
    {
        det.x1 =
            (det.x1 - prep.pad_x) / prep.scale;

        det.y1 =
            (det.y1 - prep.pad_y) / prep.scale;

        det.x2 =
            (det.x2 - prep.pad_x) / prep.scale;

        det.y2 =
            (det.y2 - prep.pad_y) / prep.scale;

        det.x1 = std::clamp(
            det.x1,
            0.0f,
            static_cast<float>(image.cols)
        );

        det.y1 = std::clamp(
            det.y1,
            0.0f,
            static_cast<float>(image.rows)
        );

        det.x2 = std::clamp(
            det.x2,
            0.0f,
            static_cast<float>(image.cols)
        );

        det.y2 = std::clamp(
            det.y2,
            0.0f,
            static_cast<float>(image.rows)
        );
    }

    // --------------------------------------------------
    // SORT BY DECENDING SCORE
    // --------------------------------------------------
    std::sort(
        final_detections.begin(),
        final_detections.end(),
        [](const Detection& a, const Detection& b)
        {
            return a.score > b.score;
        }
    );

    // --------------------------------------------------
    // PRINT FINAL COUNT TO TOP FIVE
    // --------------------------------------------------
    std::cout
        << "\nFinal detections: "
        << final_detections.size()
        << "\n";

    for (
        size_t i = 0;
        i < std::min<size_t>(5, final_detections.size());
        ++i
    )
    {
        const Detection& det =
            final_detections[i];

        std::cout
            << i + 1
            << " class=" << det.class_id
            << " score=" << det.score
            << " box=["
            << det.x1 << ", "
            << det.y1 << ", "
            << det.x2 << ", "
            << det.y2 << "]\n";
    }

    // --------------------------------------------------
    // Inputs
    // --------------------------------------------------

    const size_t inputCount =
        session.GetInputCount();

    std::cout
        << "Input count: "
        << inputCount
        << "\n";

    for (size_t i = 0; i < inputCount; ++i)
    {
        auto inputName =
            session.GetInputNameAllocated(
                i,
                allocator
            );

        auto inputTypeInfo =
            session.GetInputTypeInfo(i);

        auto tensorInfo =
            inputTypeInfo.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> shape =
            tensorInfo.GetShape();

        std::string name =
            inputName.get();

        std::cout
            << "Input "
            << i
            << "\n";

        std::cout
            << "  name: "
            << name
            << "\n";

        std::cout
            << "  shape: ";

        printShape(shape);

        std::cout
            << "\n";
    }

    // --------------------------------------------------
    // Outputs
    // --------------------------------------------------

    const size_t outputCount =
        session.GetOutputCount();

    std::cout
        << "\nOutput count: "
        << outputCount
        << "\n";

    for (size_t i = 0; i < outputCount; ++i)
    {
        auto outputName =
            session.GetOutputNameAllocated(
                i,
                allocator
            );

        auto outputTypeInfo =
            session.GetOutputTypeInfo(i);

        auto tensorInfo =
            outputTypeInfo.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> shape =
            tensorInfo.GetShape();

        std::string name =
            outputName.get();

        std::cout
            << "Output "
            << i
            << "\n";

        std::cout
            << "  name: "
            << name
            << "\n";

        std::cout
            << "  shape: ";

        printShape(shape);

        std::cout
            << "\n";
    }

    return 0;
}