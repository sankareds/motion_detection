cmake_minimum_required(VERSION 3.8)
project(detection)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
enable_language("CUDA")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
  
include_directories("/usr/local/include/opencv4" ${CUDA_INCLUDE_DIRS})
add_executable(detection detection.cu)
target_link_libraries(detection ${CUDA_LIBRARIES} ${OpenCV_LIBS})
