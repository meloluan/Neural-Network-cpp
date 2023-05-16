
cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkCpp C CXX)
set(PROJECT_DESCRIPTION   "A rock solid project with enforced style, testing and static analysis")
set(PROJECT_AUTHOR        "Luan")
set(PROJECT_URL           "https://github.com/meloluan/neural-network-cpp")
set(PROJECT_VERSION_MAJOR "0")
set(PROJECT_VERSION_MINOR "0")
set(PROJECT_VERSION_PATCH "1")

# Define as opções de compilação
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ftree-vectorize -O3 -g -fprofile-arcs -ftest-coverage")

LIST(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CmakeModules")

file(GLOB_RECURSE SOURCES "src/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "./test/")

add_library(${PROJECT_NAME}Static STATIC ${SOURCES})

target_compile_features(${PROJECT_NAME}Static PUBLIC cxx_std_17)
target_compile_options(${PROJECT_NAME}Static PRIVATE -Wall -ftree-vectorize -O3 -g -fprofile-arcs -ftest-coverage)

# Adiciona o diretório com os arquivos-fonte
add_executable(${PROJECT_NAME} src/main.cpp)

# Pass MNIST data directory to main.cpp
target_compile_definitions(${PROJECT_NAME} PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

find_package(MNIST PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

include_directories(${MNIST_INCLUDE_DIR})
target_compile_features(${PROJECT_NAME}Static PRIVATE cxx_range_for)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME}Static  Eigen3::Eigen)

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}Static)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
#   add_custom_target(check-format
#     COMMAND ./format-check.sh
#   )
  
  set(TEST_EXECUTABLE ${PROJECT_NAME}-test)
  add_subdirectory(dependencies/googletest)
  enable_testing()
  include(GoogleTest)
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories(src) # ad-hoc headers

  file(GLOB_RECURSE TESTS_SOURCES "test/*.cpp")

  add_executable(${TEST_EXECUTABLE} ${TESTS_SOURCES})
  
  gtest_add_tests(TARGET ${TEST_EXECUTABLE})
  target_link_libraries(${TEST_EXECUTABLE} ${PROJECT_NAME}Static)
  target_link_libraries(${TEST_EXECUTABLE}
    pthread
    gtest
    gmock
    gtest_main
    gmock_main
    Eigen3::Eigen
    )
  add_test(${TEST_EXECUTABLE} ${TEST_EXECUTABLE})
  
  
  if(CMAKE_COMPILER_IS_GNUCXX)
    LIST(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CmakeModules")
    include(CodeCoverage)
    APPEND_COVERAGE_COMPILER_FLAGS()
    SETUP_TARGET_FOR_COVERAGE_GCOVR_XML(NAME coverage
      EXECUTABLE ${TEST_EXECUTABLE}
      EXCLUDE "${PROJECT_SOURCE_DIR}/tests/*" "${PROJECT_SOURCE_DIR}/dependencies/*" "/usr/*" ".*/test/.*"
      BASE_DIRECTORY "${PROJECT_SOURCE_DIR}/src")
  endif()
endif()