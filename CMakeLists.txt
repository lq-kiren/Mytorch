cmake_minimum_required(VERSION 3.16)
project(mytorch LANGUAGES CXX)

add_executable(mytorch torch.cpp)
target_compile_features(mytorch PRIVATE cxx_std_17)

add_executable(backward backward.cpp)
target_compile_features(backward PRIVATE cxx_std_17)
