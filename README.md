# Aerial Object Detection: YOLO11n on VisDrone → ONNX → C++ → Raspberry Pi

An end-to-end object detection pipeline for drone imagery. I fine-tuned YOLO11n on the VisDrone dataset, evaluated it with attention to small objects, exported it to ONNX, and am building a C++ inference pipeline (OpenCV + ONNX Runtime) aimed at running on a Raspberry Pi 4.

The goal is to understand the full path from training a model to running it efficiently on edge hardware. Every stage is checked numerically against the stage before it.

---

## Project status

| Step | Description | Status |
|---|---|---|
| 1 | Baseline: pretrained YOLO11n (COCO) on VisDrone images | ✅ Done |
| 2 | Fine-tune on VisDrone (500-image subset, then full train set) | ✅ Done |
| 3 | Evaluation: precision / recall / mAP, recall by object size, failure cases | ✅ Done |
| 4 | Export to ONNX, inference in Python with ONNX Runtime, verify against Ultralytics | ✅ Done |
| 5 | C++ inference pipeline (OpenCV + ONNX Runtime + CMake) | 🔧 In progress |
| 6 | Video input and benchmarks on Raspberry Pi 4 (640 / 416 / 320) | ⏳ Planned |

---

## Pipeline

```
VisDrone images
   │
   ▼
Letterbox resize (keep aspect ratio, pad with gray 114) → RGB → /255 → CHW float tensor
   │
   ▼
YOLO11n (ONNX)  →  output [1, 14, 8400]
                     4 box values (cx, cy, w, h) + 10 class scores
                     8400 candidates = 80² + 40² + 20²  (strides 8 / 16 / 32)
   │
   ▼
Score filter (≥ 0.25) → class-aware NMS (IoU 0.7) → undo letterbox → clip to image
   │
   ▼
Detections (box, class, confidence)
```

The ONNX models are exported with static input sizes (640, 416, 320) and **without** built-in NMS, so postprocessing is implemented by hand in both Python and C++.

---

## Dataset

[VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset): drone-captured images of urban scenes, with many small and densely packed objects.

- 10 classes: `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`
- I wrote my own converter from raw VisDrone annotations (`left, top, w, h, score, category, truncation, occlusion`) to YOLO format (`class cx cy w h`, normalized). It drops "ignored regions" (category 0), "others" (category 11) and rows with score 0.
- The conversion is checked with a per-file box count, a duplicate-image hash check, and visual inspection of random images.

---

## Results

### Fine-tuning (validation set)

| Run | Images | Epochs | imgsz | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| Subset | 500 | — | 640 | 0.149 | 0.080 | — | — |
| **Full** | full train set | 50 | 640 | **0.291** | **0.165** | 0.421 | 0.321 |

Per-class AP50 varies a lot: **car 0.719**, while bicycle (0.067) and awning-tricycle (0.079) are the weakest classes.

### Recall by object size

Measured over the 38,759 validation objects, matched at IoU ≥ 0.5 and conf ≥ 0.25. Size buckets use the COCO thresholds (small < 32², medium 32²–96², large > 96² px).

| Size | Objects | Share | Found | Recall |
|---|---|---|---|---|
| Small | 26,586 | 69% | 7,373 | **28%** |
| Medium | 11,105 | 29% | 7,752 | 70% |
| Large | 1,068 | 3% | 930 | 87% |
| **All** | 38,759 | | 16,055 | **41%** |

**Main finding:** 85% of all missed objects are small. The model is usable on medium and large objects. Small objects, which make up most of VisDrone, are the main limitation at 640 px input. Experiments in step 1 showed that a higher input resolution (1280) finds more small objects, at a large compute cost. That trade-off matters on a Raspberry Pi.

> Note: this recall is weighted by object count, whereas Ultralytics reports recall averaged per class. That's why the two numbers differ.

### Failure cases

I studied the images with the lowest recall:

| Image | Recall | Observation |
|---|---|---|
| `0000276_05601_d_0000535` | 0% | Daylight scene with hard shadows |
| `0000291_00001_d_0000868` | 7.9% | Very dense, very small objects |
| `0000001_04527_d_0000008` | 12.2% | Small, distant objects |
| `0000242_06010_d_0000017` | 12.2% | Small, distant objects |

### ONNX vs Ultralytics (correctness check)

My own Python preprocessing and postprocessing (NumPy + ONNX Runtime) compared with Ultralytics on the same image:

| Check | Result |
|---|---|
| Detections | 26 / 26 matched |
| Lowest / average IoU | 1.0 / 1.0 |
| Largest confidence difference | 0.0 |
| Sanity check: boxes shifted by 5 px | 17 / 26 matched, lowest IoU 0.51 (the check does catch errors) |

### Python ONNX Runtime speed (laptop CPU, Windows, median)

| Input size | Preprocess | Inference | Postprocess | Total | ≈ FPS | Detections |
|---|---|---|---|---|---|---|
| 640 | 13.3 ms | 70.4 ms | 4.8 ms | 88.8 ms | 11 | 26 |
| 416 | 6.3 ms | 34.2 ms | 4.6 ms | 45.6 ms | 22 | 24 |
| 320 | 3.5 ms | 16.2 ms | 1.7 ms | 21.2 ms | 46 | 19 |

Inference is about 80% of the total time. A C++ rewrite can only speed up pre- and postprocessing. Smaller input sizes are much faster but find fewer objects, mostly small ones.

### C++ pipeline (step 5, in progress)

- ✅ Load the image and the ONNX model, read input/output names and shapes
- ✅ Letterbox preprocessing matches Python: same scale, padding and pixel values, tensor checksum within 0.02 of 552,103 (≈ 4×10⁻⁸ relative)
- ⏳ Run inference and read the raw output
- ⏳ Postprocessing and NMS (target: the same 26 detections as Python)
- ⏳ Detector class, drawing, CLI, C++ vs Python benchmark

---

## Repository structure

```
.
├── cpp/
│   ├── CMakeLists.txt
│   └── src/main.cpp            # C++ inference pipeline (OpenCV + ONNX Runtime)
├── python/
│   ├── onnx_infer.py           # preprocess / infer / postprocess / NMS in NumPy
│   ├── verify_against_ultralytics.py
│   └── benchmark.py            # timing at 640 / 416 / 320
├── models/                     # visdrone_yolo11n_{640,416,320}.onnx
├── image/                      # test images
└── third_party/                # ONNX Runtime (not committed)
```

---

## Running it

### Python

```bash
python -m venv .venv && source .venv/bin/activate
pip install onnxruntime opencv-python numpy ultralytics

python python/onnx_infer.py
python python/verify_against_ultralytics.py
python python/benchmark.py
```

### C++ (Ubuntu 24.04 / WSL2)

Requirements: gcc 13, CMake, pkg-config, `libopencv-dev`, and [ONNX Runtime 1.31.0](https://github.com/microsoft/onnxruntime/releases) (`onnxruntime-linux-x64-1.31.0`) unpacked into `third_party/`.

```bash
sudo apt install build-essential cmake pkg-config libopencv-dev

cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build cpp/build
./cpp/build/<executable-name>
```

The ONNX Runtime library path is embedded in the binary through RPATH, so `LD_LIBRARY_PATH` isn't needed.

---

## Tools

Python · Ultralytics YOLO11 (8.4) · PyTorch · Google Colab (T4) · ONNX · ONNX Runtime · NumPy · OpenCV · C++17 · CMake · WSL2 · Raspberry Pi 4 (planned target)

## Next steps

- Finish the C++ pipeline and compare its speed with Python
- Video input
- Benchmark 640 / 416 / 320 on a Raspberry Pi 4 (64-bit OS)
- Investigate improving small-object recall: higher input resolution, tiling/slicing, or a P2 detection head

## Author

Justus Anoruo · [github.com/justusa11](https://github.com/justusa11)