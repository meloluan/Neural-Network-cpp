
cmake_minimum_required(VERSION 3.10)
project(NeuralNetworkCpp C CXX)
set(PROJECT_DESCRIPTION   "A library for neural network implementation")
set(PROJECT_AUTHOR        "Luan Melo")
set(PROJECT_URL           "https://github.com/meloluan/neural-network-cpp")
set(PROJECT_VERSION_MAJOR "1")
set(PROJECT_VERSION_MINOR "0")
set(PROJECT_VERSION_PATCH "0")

include(${CMAKE_CURRENT_SOURCE_DIR}/doc/doxygen.cmake)

# Define a target para a geração da documentação com o Doxygen
add_custom_target(documentation
    COMMAND doxygen ${CMAKE_CURRENT_SOURCE_DIR}/doc/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc/
    COMMENT "Generating documentation with Doxygen"
)

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

find_package(MNIST PATHS ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mnist)
if(NOT MNIST_FOUND)
    message(FATAL_ERROR "MNIST loader could not be found. It is available under https://github.com/wichtounet/mnist")
endif(NOT MNIST_FOUND)

set(MNIST_DATA_DIR ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/mnist)

# Pass MNIST data directory to main.cpp
target_compile_definitions(${PROJECT_NAME}Static PUBLIC MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

include_directories(${MNIST_INCLUDE_DIR})
target_compile_features(${PROJECT_NAME}Static PRIVATE cxx_range_for)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

target_link_libraries(${PROJECT_NAME}Static  Eigen3::Eigen)

find_package(PythonLibs 3)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dependencies/matplotlib-cpp)
target_include_directories(${PROJECT_NAME}Static PUBLIC ${PYTHON_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/dependencies/matplotlib-cpp)
target_link_libraries(${PROJECT_NAME}Static ${PYTHON_LIBRARIES})

target_link_libraries(${PROJECT_NAME} PUBLIC ${PROJECT_NAME}Static)


if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  
  set(TEST_EXECUTABLE ${PROJECT_NAME}Test)
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

set_target_properties( ${PROJECT_NAME} ${PROJECT_NAME}Static ${TEST_EXECUTABLE}
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
