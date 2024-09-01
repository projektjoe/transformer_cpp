cmake_minimum_required(VERSION 3.22)
project(transformer_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(transformer_cpp
        test.cpp
        main.cpp
        tokenizer.cpp tokenizer.h)
