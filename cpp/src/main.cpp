#include "detector.hpp"

#include <opencv2/opencv.hpp>

#include <array>
#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr
            << "Usage: "
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg> [conf]\n";

        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];

    float conf_threshold = 0.25f;

    if (argc >= 5)
    {
        conf_threshold = std::stof(argv[4]);
    }

    cv::Mat image = cv::imread(input_path);

    if (image.empty())
    {
        std::cerr
            << "Could not load image: "
            << input_path
            << "\n";

        return 1;
    }

    Detector detector(model_path);

    std::vector<Detection> detections =
        detector.detect(
            image,
            conf_threshold,
            0.7f
        );

    std::cout
        << "Detections: "
        << detections.size()
        << "\n";

    return 0;
}