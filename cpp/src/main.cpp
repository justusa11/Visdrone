#include "detector.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>


double median(std::vector<double> values)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 1)
    {
        return values[n / 2];
    }

    return (
        values[n / 2 - 1]
        + values[n / 2]
    ) / 2.0;
}


int main(int argc, char* argv[])
{
    if (argc < 4)
    {
        std::cerr
            << "Usage:\n"
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg> [conf]\n"
            << "\nBenchmark:\n"
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg> --bench <runs>\n";

        return 1;
    }

    const std::string model_path = argv[1];
    const std::string input_path = argv[2];
    const std::string output_path = argv[3];

    float conf_threshold = 0.25f;

    bool benchmark_mode = false;
    int benchmark_runs = 50;

    // -----------------------------------------------
    // Parse optional arguments
    // -----------------------------------------------

    if (argc >= 5)
    {
        const std::string arg4 = argv[4];

        if (arg4 == "--bench")
        {
            benchmark_mode = true;

            if (argc >= 6)
            {
                benchmark_runs = std::stoi(argv[5]);
            }
        }
        else
        {
            conf_threshold = std::stof(arg4);
        }
    }

    // -----------------------------------------------
    // Load image once
    // -----------------------------------------------

    cv::Mat image = cv::imread(input_path);

    if (image.empty())
    {
        std::cerr
            << "Could not load image: "
            << input_path
            << "\n";

        return 1;
    }

    // -----------------------------------------------
    // Create detector once
    // -----------------------------------------------

    Detector detector(model_path);


    // ===============================================
    // BENCHMARK MODE
    // ===============================================

    if (benchmark_mode)
    {
        constexpr int WARMUP_RUNS = 3;

        std::cout
            << "Benchmark mode\n"
            << "Warmup runs: "
            << WARMUP_RUNS
            << "\n"
            << "Timed runs: "
            << benchmark_runs
            << "\n\n";

        // Warmup
        for (int i = 0; i < WARMUP_RUNS; ++i)
        {
            detector.detect(
                image,
                conf_threshold,
                0.7f
            );
        }

        std::vector<double> pre_times;
        std::vector<double> infer_times;
        std::vector<double> post_times;
        std::vector<double> total_times;

        pre_times.reserve(benchmark_runs);
        infer_times.reserve(benchmark_runs);
        post_times.reserve(benchmark_runs);
        total_times.reserve(benchmark_runs);

        int detection_count = 0;

        for (int i = 0; i < benchmark_runs; ++i)
        {
            std::vector<Detection> detections =
                detector.detect(
                    image,
                    conf_threshold,
                    0.7f
                );

            const Timings& t =
                detector.last_timings();

            pre_times.push_back(
                t.pre_ms
            );

            infer_times.push_back(
                t.infer_ms
            );

            post_times.push_back(
                t.post_ms
            );

            total_times.push_back(
                t.pre_ms
                + t.infer_ms
                + t.post_ms
            );

            detection_count =
                static_cast<int>(
                    detections.size()
                );
        }

        const double pre_median =
            median(pre_times);

        const double infer_median =
            median(infer_times);

        const double post_median =
            median(post_times);

        const double total_median =
            median(total_times);

        const double fps =
            total_median > 0.0
            ? 1000.0 / total_median
            : 0.0;

        std::cout
            << "Detections: "
            << detection_count
            << "\n\n";

        std::cout
            << "Median preprocess: "
            << pre_median
            << " ms\n";

        std::cout
            << "Median inference: "
            << infer_median
            << " ms\n";

        std::cout
            << "Median postprocess: "
            << post_median
            << " ms\n";

        std::cout
            << "Median total: "
            << total_median
            << " ms\n";

        std::cout
            << "FPS: "
            << fps
            << "\n";

        return 0;
    }


    // ===============================================
    // NORMAL IMAGE MODE
    // ===============================================

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

    const std::array<std::string, 10> class_names = {
        "pedestrian",
        "people",
        "bicycle",
        "car",
        "van",
        "truck",
        "tricycle",
        "awning-tricycle",
        "bus",
        "motor"
    };

    for (const Detection& det : detections)
    {
        const int x1 =
            static_cast<int>(det.x1);

        const int y1 =
            static_cast<int>(det.y1);

        const int x2 =
            static_cast<int>(det.x2);

        const int y2 =
            static_cast<int>(det.y2);

        const cv::Scalar color(
            0,
            255,
            0
        );

        cv::rectangle(
            image,
            cv::Point(x1, y1),
            cv::Point(x2, y2),
            color,
            2
        );

        std::string label =
            class_names[det.class_id]
            + " "
            + cv::format(
                "%.2f",
                det.score
            );

        cv::putText(
            image,
            label,
            cv::Point(
                x1,
                std::max(
                    10,
                    y1 - 5
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv::LINE_AA
        );
    }

    if (!cv::imwrite(
        output_path,
        image
    ))
    {
        std::cerr
            << "Failed to save output image: "
            << output_path
            << "\n";

        return 1;
    }

    std::cout
        << "Saved: "
        << output_path
        << "\n";

    return 0;
}