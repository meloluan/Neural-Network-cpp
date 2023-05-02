# neural-network-cpp
# A rock solid project with enforced style, testing and static analysis
# Copyright 2023, Luan
cmake_minimum_required(VERSION 3.10)
project(neural-network-cpp C CXX)
set(PROJECT_DESCRIPTION   "A rock solid project with enforced style, testing and static analysis")
set(PROJECT_AUTHOR        "Luan")
set(PROJECT_URL           "https://github.com/meloluan/neural-network-cpp")
set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "0")
set(PROJECT_VERSION_PATCH "1")

# Define as opções de compilação
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ftree-vectorize -O3 -Ofast")

file(GLOB_RECURSE SOURCES "src/*.cpp")
set(SOURCES_COPY ${SOURCES})

# Adiciona o diretório com os arquivos-fonte
add_executable(neural-network-cpp ${SOURCES_COPY})

find_package(MNIST PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})
target_compile_features(neural-network-cpp PRIVATE cxx_range_for)

# Pass MNIST data directory to main.cpp
target_compile_definitions(neural-network-cpp PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

# Define as bibliotecas necessárias
# find_package(OpenCV REQUIRED )        
# INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

 find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(neural-network-cpp  Eigen3::Eigen)