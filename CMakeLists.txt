cmake_minimum_required(VERSION 3.18)
project(edge_ai LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_executable(edge_ai main.cpp postprocess.cu preprocess.cu)
target_include_directories(edge_ai PRIVATE ${OpenCV_INCLUDE_DIRS} /usr/include/x86_64-linux-gnu)
target_link_libraries(edge_ai PRIVATE ${OpenCV_LIBS} nvinfer CUDA::cudart)
