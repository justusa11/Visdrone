# Visdrone

A small C++/OpenCV project for working with VisDrone images. The current
executable loads an image and prints its width, height, and channel count.

## Requirements

- Linux or WSL
- CMake 3.16 or newer
- A C++17 compiler
- OpenCV development libraries

On Ubuntu/WSL, install the build requirements with:

```bash
sudo apt update
sudo apt install build-essential cmake libopencv-dev
```

## Build

From the repository root:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

## Run

Pass an image path to the executable:

```bash
./cpp/build/visdrone_app image/0000001_04527_d_0000008.jpg
```

Expected output resembles:

```text
Width: 1920
