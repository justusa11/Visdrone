cmake_minimum_required(VERSION 3.16)

project(visdrone_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

get_filename_component(
    ONNXRUNTIME_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/onnxruntime-linux-x64-1.31.0"
    ABSOLUTE
 )

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(
    onnxruntime
    PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include"
)

add_executable(
    visdrone_app 
    main.cpp
)

target_link_libraries(
    visdrone_app
    PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)