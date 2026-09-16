cmake_minimum_required(VERSION 3.16)
project(mytorch LANGUAGES CXX)

add_library(mytorch src/tensor.cpp)
target_include_directories(mytorch PUBLIC include)
target_compile_features(mytorch PUBLIC cxx_std_17)

enable_testing()

add_executable(tensor_test tests/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE mytorch)
add_test(NAME tensor_test COMMAND tensor_test)

add_executable(forward_ad experiments/scalar_ad/forward.cpp)
target_compile_features(forward_ad PRIVATE cxx_std_17)

add_executable(reverse_ad experiments/scalar_ad/reverse.cpp)
target_compile_features(reverse_ad PRIVATE cxx_std_17)
