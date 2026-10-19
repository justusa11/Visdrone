cmake_minimum_required(VERSION 3.16)

project(visdrone_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------
# OpenCV
# --------------------------------------------------

find_package(OpenCV REQUIRED)

# --------------------------------------------------
# ONNX Runtime
# --------------------------------------------------

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

set(
    ONNXRUNTIME_DIR
    "${CMAKE_SOURCE_DIR}/../third_party/onnxruntime-linux-x64-1.31.0"
)

# --------------------------------------------------
# Executable
# --------------------------------------------------

add_executable(
    visdrone_app
    src/main.cpp
    src/detector.cpp
)

# --------------------------------------------------
# Include directories
# --------------------------------------------------

target_include_directories(
    visdrone_app
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${ONNXRUNTIME_DIR}/include"
)

# --------------------------------------------------
# Libraries
# --------------------------------------------------

target_link_libraries(
    visdrone_app
    PRIVATE
    ${OpenCV_LIBS}
    "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
)

# --------------------------------------------------
# Runtime path for libonnxruntime.so
# --------------------------------------------------

set_target_properties(
    visdrone_app
    PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_DIR}/lib"
)