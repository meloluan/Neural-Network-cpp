# CMake generated Testfile for 
# Source directory: /home/brabo/Projetos/neural-network-cpp
# Build directory: /home/brabo/Projetos/neural-network-cpp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(PCATest.DimensionalityReduction "/home/brabo/Projetos/neural-network-cpp/neural-network-cpp-test" "--gtest_filter=PCATest.DimensionalityReduction")
set_tests_properties(PCATest.DimensionalityReduction PROPERTIES  SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]" _BACKTRACE_TRIPLES "/home/brabo/.local/lib/python3.10/site-packages/cmake/data/share/cmake-3.26/Modules/GoogleTest.cmake;402;add_test;/home/brabo/Projetos/neural-network-cpp/CMakeLists.txt;62;gtest_add_tests;/home/brabo/Projetos/neural-network-cpp/CMakeLists.txt;0;")
add_test(neural-network-cpp-test "neural-network-cpp-test")
set_tests_properties(neural-network-cpp-test PROPERTIES  _BACKTRACE_TRIPLES "/home/brabo/Projetos/neural-network-cpp/CMakeLists.txt;72;add_test;/home/brabo/Projetos/neural-network-cpp/CMakeLists.txt;0;")
subdirs("dependencies/googletest")
