#include "detector.hpp"

#include <opencv2/opencv.hpp>

#include <chrono>
#include <cmath>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// --------------------------------------------------
// Class names
// --------------------------------------------------

const std::array<std::string, 10> CLASS_NAMES = {
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


// --------------------------------------------------
// Median helper for benchmark mode
// --------------------------------------------------

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

// PERCENTILE HELPER
double percentile(
    std::vector<double> values,
    double p
)
{
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(
        values.begin(),
        values.end()
    );

    const double index =
        p * (values.size() - 1);

    const size_t lower =
        static_cast<size_t>(
            std::floor(index)
        );

    const size_t upper =
        static_cast<size_t>(
            std::ceil(index)
        );

    if (lower == upper)
    {
        return values[lower];
    }

    const double weight =
        index - lower;

    return
        values[lower] * (1.0 - weight)
        + values[upper] * weight;
}

// --------------------------------------------------
// Drawing helper
// Used by BOTH image and video mode
// --------------------------------------------------

void draw_detections(
    cv::Mat& image,
    const std::vector<Detection>& detections
)
{
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

        std::string class_name = "unknown";

        if (
            det.class_id >= 0
            && det.class_id < static_cast<int>(
                CLASS_NAMES.size()
            )
        )
        {
            class_name =
                CLASS_NAMES[det.class_id];
        }

        std::string label =
            class_name
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
                    15,
                    y1 - 5
                )
            ),
            cv::FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv::LINE_AA
        );
    }
}


// --------------------------------------------------
// Main
// --------------------------------------------------

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cerr
            << "Usage:\n\n"

            << "Image mode:\n"
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg> [conf]\n\n"

            << "Benchmark mode:\n"
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg> --bench <runs>\n\n"

            << "Video mode:\n"
            << argv[0]
            << " <model.onnx> --video <input.mp4> <output.mp4> [conf]\n";

        return 1;
    }

    const std::string model_path =
        argv[1];


    // ==================================================
    // VIDEO MODE
    // ==================================================

    if (
        argc >= 5
        && std::string(argv[2]) == "--video"
    )
    {
        const std::string input_video =
            argv[3];

        const std::string output_video =
            argv[4];

        float conf_threshold = 0.25f;

        if (argc >= 6)
        {
            conf_threshold =
                std::stof(argv[5]);
        }

        // ----------------------------------------------
        // Open input video
        // ----------------------------------------------

        cv::VideoCapture cap(
            input_video
        );

        if (!cap.isOpened())
        {
            std::cerr
                << "Could not open video: "
                << input_video
                << "\n";

            return 1;
        }

        const double fps =
            cap.get(
                cv::CAP_PROP_FPS
            );

        const int width =
            static_cast<int>(
                cap.get(
                    cv::CAP_PROP_FRAME_WIDTH
                )
            );

        const int height =
            static_cast<int>(
                cap.get(
                    cv::CAP_PROP_FRAME_HEIGHT
                )
            );

        const int total_frames =
            static_cast<int>(
                cap.get(
                    cv::CAP_PROP_FRAME_COUNT
                )
            );

        std::cout
            << "Video opened\n"
            << "FPS: "
            << fps
            << "\n"
            << "Resolution: "
            << width
            << "x"
            << height
            << "\n"
            << "Frames: "
            << total_frames
            << "\n\n";


        // ----------------------------------------------
        // Open output video
        // ----------------------------------------------

        const int fourcc =
            cv::VideoWriter::fourcc(
                'm',
                'p',
                '4',
                'v'
            );

        cv::VideoWriter writer(
            output_video,
            fourcc,
            fps,
            cv::Size(
                width,
                height
            )
        );

        if (!writer.isOpened())
        {
            std::cerr
                << "Could not create output video: "
                << output_video
                << "\n";

            return 1;
        }


        // ----------------------------------------------
        // Create detector ONCE
        // ----------------------------------------------

        Detector detector(
            model_path
        );

        cv::Mat frame;

        int frame_number = 0;

        // Timing Vectors
        constexpr int VIDEO_WARMUP_FRAMES = 5;

        std::vector<double> read_times;
        std::vector<double> pre_times;
        std::vector<double> infer_times;
        std::vector<double> post_times;
        std::vector<double> draw_times;
        std::vector<double> write_times;
        std::vector<double> detect_times;
        std::vector<double> end_to_end_times;

        // ----------------------------------------------
        // Video loop
        // ----------------------------------------------

        while (true)
        {
            // ------------------------------------------
            // READ
            // ------------------------------------------

            auto read_start =
                std::chrono::steady_clock::now();

            const bool ok =
                cap.read(frame);

            auto read_end =
                std::chrono::steady_clock::now();

            if (!ok || frame.empty())
            {
                break;
            }

            const double read_ms =
                std::chrono::duration<double, std::milli>(
                    read_end - read_start
                ).count();


            // ------------------------------------------
            // DETECT
            // ------------------------------------------

            std::vector<Detection> detections =
                detector.detect(
                    frame,
                    conf_threshold,
                    0.7f
                );

            const Timings& timings =
                detector.last_timings();

            const double pre_ms =
                timings.pre_ms;

            const double infer_ms =
                timings.infer_ms;

            const double post_ms =
                timings.post_ms;

            const double detect_ms =
                pre_ms
                + infer_ms
                + post_ms;


            // ------------------------------------------
            // DRAW
            // ------------------------------------------

            auto draw_start =
                std::chrono::steady_clock::now();

            draw_detections(
                frame,
                detections
            );

            const double detector_fps =
                detect_ms > 0.0
                ? 1000.0 / detect_ms
                : 0.0;

            std::ostringstream overlay;

            overlay
                << std::fixed
                << std::setprecision(1)
                << "FPS: "
                << detector_fps
                << " | "
                << detect_ms
                << " ms | "
                << detections.size()
                << " det";

            cv::putText(
                frame,
                overlay.str(),
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 0),
                2,
                cv::LINE_AA
            );

            auto draw_end =
                std::chrono::steady_clock::now();

            const double draw_ms =
                std::chrono::duration<double, std::milli>(
                    draw_end - draw_start
                ).count();


            // ------------------------------------------
            // WRITE
            // ------------------------------------------

            auto write_start =
                std::chrono::steady_clock::now();

            writer.write(frame);

            auto write_end =
                std::chrono::steady_clock::now();

            const double write_ms =
                std::chrono::duration<double, std::milli>(
                    write_end - write_start
                ).count();


            // ------------------------------------------
            // STORE STATISTICS
            // Skip first 5 warmup frames
            // ------------------------------------------

            if (frame_number >= VIDEO_WARMUP_FRAMES)
            {
                const double end_to_end_ms =
                    read_ms
                    + detect_ms
                    + draw_ms
                    + write_ms;

                read_times.push_back(read_ms);
                pre_times.push_back(pre_ms);
                infer_times.push_back(infer_ms);
                post_times.push_back(post_ms);
                draw_times.push_back(draw_ms);
                write_times.push_back(write_ms);

                detect_times.push_back(
                    detect_ms
                );

                end_to_end_times.push_back(
                    end_to_end_ms
                );
            }


            ++frame_number;


            // ------------------------------------------
            // Progress
            // ------------------------------------------

            if (frame_number % 100 == 0)
            {
                std::cout
                    << "Processed "
                    << frame_number
                    << " / "
                    << total_frames
                    << " frames\n";
            }
        }


        // ----------------------------------------------
        // Cleanup
        // ----------------------------------------------

        cap.release();
        writer.release();

        // PRINT STATISTICS
        auto print_stats =
            [](
                const std::string& name,
                const std::vector<double>& values
            )
        {
            std::cout
                << name
                << "\n"
                << "  Median: "
                << median(values)
                << " ms\n"
                << "  P95:    "
                << percentile(values, 0.95)
                << " ms\n"
                << "  Max:    "
                << *std::max_element(
                    values.begin(),
                    values.end()
                )
                << " ms\n";
        };

        std::cout
            << "\n===== VIDEO TIMING =====\n";

        print_stats(
            "Read",
            read_times
        );

        print_stats(
            "Preprocess",
            pre_times
        );

        print_stats(
            "Inference",
            infer_times
        );

        print_stats(
            "Postprocess",
            post_times
        );

        print_stats(
            "Draw",
            draw_times
        );

        print_stats(
            "Write",
            write_times
        );

        print_stats(
            "Detect total",
            detect_times
        );

        print_stats(
            "End-to-end",
            end_to_end_times
        );

        // FPS VALUES
        const double median_detect =
            median(detect_times);

        const double median_e2e =
            median(end_to_end_times);

        const double detector_fps =
            median_detect > 0.0
            ? 1000.0 / median_detect
            : 0.0;

        const double end_to_end_fps =
            median_e2e > 0.0
            ? 1000.0 / median_e2e
            : 0.0;

        std::cout
            << "\nDetector FPS: "
            << detector_fps
            << "\n"
            << "End-to-end FPS: "
            << end_to_end_fps
            << "\n";

        std::cout
            << "\nFinished.\n"
            << "Processed frames: "
            << frame_number
            << "\n"
            << "Saved: "
            << output_video
            << "\n";

        return 0;
    }


    // ==================================================
    // IMAGE / BENCHMARK MODE
    // ==================================================

    if (argc < 4)
    {
        std::cerr
            << "Image mode requires:\n"
            << argv[0]
            << " <model.onnx> <input.jpg> <output.jpg>\n";

        return 1;
    }

    const std::string input_path =
        argv[2];

    const std::string output_path =
        argv[3];

    float conf_threshold =
        0.25f;

    bool benchmark_mode =
        false;

    int benchmark_runs =
        50;


    // --------------------------------------------------
    // Parse image/benchmark arguments
    // --------------------------------------------------

    if (argc >= 5)
    {
        const std::string arg4 =
            argv[4];

        if (arg4 == "--bench")
        {
            benchmark_mode = true;

            if (argc >= 6)
            {
                benchmark_runs =
                    std::stoi(
                        argv[5]
                    );
            }
        }
        else
        {
            conf_threshold =
                std::stof(
                    arg4
                );
        }
    }


    // --------------------------------------------------
    // Load image
    // --------------------------------------------------

    cv::Mat image =
        cv::imread(
            input_path
        );

    if (image.empty())
    {
        std::cerr
            << "Could not load image: "
            << input_path
            << "\n";

        return 1;
    }


    // --------------------------------------------------
    // Create detector
    // --------------------------------------------------

    Detector detector(
        model_path
    );


    // ==================================================
    // BENCHMARK MODE
    // ==================================================

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

        for (
            int i = 0;
            i < WARMUP_RUNS;
            ++i
        )
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

        pre_times.reserve(
            benchmark_runs
        );

        infer_times.reserve(
            benchmark_runs
        );

        post_times.reserve(
            benchmark_runs
        );

        total_times.reserve(
            benchmark_runs
        );

        int detection_count = 0;

        for (
            int i = 0;
            i < benchmark_runs;
            ++i
        )
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
            median(
                pre_times
            );

        const double infer_median =
            median(
                infer_times
            );

        const double post_median =
            median(
                post_times
            );

        const double total_median =
            median(
                total_times
            );

        const double benchmark_fps =
            total_median > 0.0
            ? 1000.0 / total_median
            : 0.0;

        std::cout
            << "Detections: "
            << detection_count
            << "\n\n"

            << "Median preprocess: "
            << pre_median
            << " ms\n"

            << "Median inference: "
            << infer_median
            << " ms\n"

            << "Median postprocess: "
            << post_median
            << " ms\n"

            << "Median total: "
            << total_median
            << " ms\n"

            << "FPS: "
            << benchmark_fps
            << "\n";

        return 0;
    }


    // ==================================================
    // NORMAL IMAGE MODE
    // ==================================================

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

    draw_detections(
        image,
        detections
    );

    if (
        !cv::imwrite(
            output_path,
            image
        )
    )
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