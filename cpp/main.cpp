#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>

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