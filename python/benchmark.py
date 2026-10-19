# LOAD LIBRARIES
import cv2
import time
import numpy as np
import onnxruntime as ort

from inspect_onnx import (
    preprocess,
    infer,
    postprocess
)

# MODEL CONFIGURATIONS
MODEL_CONFIGS = [
    ("models/visdrone_yolo11n_640.onnx", 640),
    ("models/visdrone_yolo11n_416.onnx", 416),
    ("models/visdrone_yolo11n_320.onnx", 320),
]

IMAGE_PATH = "image/0000242_02400_d_0000009.jpg"

WARMUP_RUNS = 3
TIMED_RUNS = 50


def summarize(times_ms):
    return {
        "mean": float(np.mean(times_ms)),
        "median": float(np.median(times_ms)),
    }


for model_path, input_size in MODEL_CONFIGS:

    print()
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input size: {input_size}")
    print("=" * 60)

    # Load image
    
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise ValueError(
            f"Could not load image: {IMAGE_PATH}"
        )

    orig_h, orig_w = image.shape[:2]


    # Create ONNX session

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"]
    )

    # Warm-up

    for _ in range(WARMUP_RUNS):

        tensor, scale, pad_x, pad_y = preprocess(
            image,
            input_size=input_size
        )

        raw_output = infer(
            session,
            tensor
        )

        detections = postprocess(
            raw_output,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
            conf_threshold=0.25,
            iou_threshold=0.7
        )

    # Timed runs

    preprocess_times = []
    inference_times = []
    postprocess_times = []
    total_times = []

    detection_count = 0

    for _ in range(TIMED_RUNS):

        #total_start = time.perf_counter()

        # Preprocess
        start = time.perf_counter()

        tensor, scale, pad_x, pad_y = preprocess(
            image,
            input_size=input_size
        )

        end = time.perf_counter()

        pre_ms = (end - start) * 1000
        preprocess_times.append(pre_ms)
        
        # Inference
        start = time.perf_counter()

        raw_output = infer(
            session,
            tensor
        )

        end = time.perf_counter()

        infer_ms = (end - start) * 1000

        inference_times.append(infer_ms)

        # Postprocess
        start = time.perf_counter()

        detections = postprocess(
            raw_output,
            scale,
            pad_x,
            pad_y,
            orig_w,
            orig_h,
            conf_threshold=0.25,
            iou_threshold=0.7
        )

        end = time.perf_counter()

        post_ms = (end - start) * 1000

        postprocess_times.append(post_ms)

        detection_count = len(detections)

        # Total
        total_ms = pre_ms + infer_ms + post_ms

        total_times.append(total_ms)

    # --------------------------------------------------
    # Summaries
    # --------------------------------------------------

    prep_stats = summarize(preprocess_times)
    infer_stats = summarize(inference_times)
    post_stats = summarize(postprocess_times)
    total_stats = summarize(total_times)

    fps = 1000.0 / total_stats["median"]

    print()
    print(f"Detections: {detection_count}")

    print()
    print("Preprocess")
    print(f"  Mean:   {prep_stats['mean']:.2f} ms")
    print(f"  Median: {prep_stats['median']:.2f} ms")

    print("Inference")
    print(f"  Mean:   {infer_stats['mean']:.2f} ms")
    print(f"  Median: {infer_stats['median']:.2f} ms")

    print("Postprocess")
    print(f"  Mean:   {post_stats['mean']:.2f} ms")
    print(f"  Median: {post_stats['median']:.2f} ms")

    print("Total")
    print(f"  Mean:   {total_stats['mean']:.2f} ms")
    print(f"  Median: {total_stats['median']:.2f} ms")

    print()
    print(f"FPS: {fps:.2f}")