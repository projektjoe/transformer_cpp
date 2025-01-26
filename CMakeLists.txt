cmake_minimum_required(VERSION 3.22)
project(transformer_cpp)
project(test_transformer_cpp)
set(CMAKE_CXX_STANDARD 17)


add_executable(transformer_cpp
        main.cpp
        tokenizer.cpp tokenizer.h)

add_executable(test_transformer_cpp
        test.cpp
        tokenizer.cpp tokenizer.h)
