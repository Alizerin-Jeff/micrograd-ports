cmake_minimum_required(VERSION 3.15)
project(MicrogradClone LANGUAGES CXX)

# Require C++23 (or higher if you want)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(micrograd
    main.cpp
    engine.cpp
    model/neuron.cpp
    model/layer.cpp
    model/mlp.cpp
)

# Compiler warnings
target_compile_options(micrograd PRIVATE -Wall -Wextra -Wpedantic)