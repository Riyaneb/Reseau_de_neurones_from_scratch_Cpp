cmake_minimum_required(VERSION 3.22)

project(ReseauDeNeuroneFromScratch LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    find_package(CUDAToolkit REQUIRED)
endif()


set(CORE_SOURCE
        src/env/CorridorEnv.cpp
        src/env/SnakeEnv.cpp
        src/io/Serializer.cpp
        src/math/Matrix.cpp
        src/math/Random.cpp
        src/math/Types.cpp
        src/nn/ActivationLayer.cpp
        src/nn/Activations.cpp
        src/nn/DenseLayer.cpp
        src/nn/Initializers.cpp
        src/nn/Layer.cpp
        src/nn/Losses.cpp
        src/nn/Network.cpp
        src/nn/Optimizers.cpp
        src/rl/DQNAgent.cpp
        src/rl/EpsilonSchedule.cpp
        src/rl/ReplayBuffer.cpp
        src/utils/Timer.cpp
        src/utils/Logger.cpp
        include/nn/SoftmaxLayer.hpp
        src/nn/SoftmaxLayer.cpp
        include/gpu/GpuCudaCheck.hpp
        include/gpu/GpuCudaCheck.hpp
        include/gpu/Gpu_calcul_matrix.hpp
        include/nn/Shape.hpp
        src/nn/Shape.cpp
        include/cnn/FlattenLayer.hpp
        src/cnn/FlattenLayer.cpp
        src/cnn/ConvLayer.cpp
        include/cnn/ConvLayer.hpp
        include/cnn/MaxPoolingLayer.hpp
        src/cnn/MaxPoolingLayer.cpp
        include/utils/DatasetImage.hpp
        src/utils/DatasetImage.cpp
)

set(TEST
        tests/main.cpp
        tests/test_gradient_check.cpp
        tests/test_layers.cpp
        tests/test_matrix.cpp
        tests/test_replay_buffer.cpp
        tests/test_snake_env.cpp
        tests/test_random.cpp
        tests/test_corridor_env.cpp
)

add_library(libNeurone ${CORE_SOURCE}
        include/utils/Logger.hpp)
target_compile_features(libNeurone PUBLIC cxx_std_17)
target_include_directories(libNeurone PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(libNeurone PUBLIC DATASET_DIR="${CMAKE_SOURCE_DIR}/data/dataset_images")

if(CMAKE_CUDA_COMPILER)
    target_sources(libNeurone PRIVATE src/gpu/Gpu_calcul_matrix.cu)
    target_compile_definitions(libNeurone PUBLIC USE_CUDA)
    target_link_libraries(libNeurone PUBLIC CUDA::cudart)
    target_link_libraries(libNeurone PUBLIC CUDA::cublas)
endif()


add_executable(play_snake apps/play_snake.cpp)
target_link_libraries(play_snake PRIVATE libNeurone)
target_compile_features(play_snake PRIVATE cxx_std_17)
set_target_properties(play_snake PROPERTIES CXX_EXTENSIONS OFF)

add_executable(train_snake apps/train_snake.cpp)
target_link_libraries(train_snake PRIVATE libNeurone)
target_compile_features(train_snake PRIVATE cxx_std_17)
set_target_properties(train_snake PROPERTIES CXX_EXTENSIONS OFF)

add_executable(train_xor apps/train_xor.cpp)
target_link_libraries(train_xor PRIVATE libNeurone)
target_compile_features(train_xor PRIVATE cxx_std_17)
set_target_properties(train_xor PROPERTIES CXX_EXTENSIONS OFF)

add_executable(benchmark_matrix apps/benchmark_matrix.cpp)
target_link_libraries(benchmark_matrix PRIVATE libNeurone)
target_compile_features(benchmark_matrix PRIVATE cxx_std_17)
set_target_properties(benchmark_matrix PROPERTIES CXX_EXTENSIONS OFF)

add_executable(train_corridor apps/train_corridor.cpp)
target_link_libraries(train_corridor PRIVATE libNeurone)
target_compile_features(train_corridor PRIVATE cxx_std_17)
set_target_properties(train_corridor PROPERTIES CXX_EXTENSIONS OFF)

add_executable(agent_snake apps/agent_snake.cpp)
target_link_libraries(agent_snake PRIVATE libNeurone)
target_compile_features(agent_snake PRIVATE cxx_std_17)
set_target_properties(agent_snake PROPERTIES CXX_EXTENSIONS OFF)

add_executable(train_mnist apps/train_mnist.cpp)
target_link_libraries(train_mnist PRIVATE libNeurone)
target_compile_features(train_mnist PRIVATE cxx_std_17)
set_target_properties(train_mnist PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_mnist apps/test_mnist.cpp)
target_link_libraries(test_mnist PRIVATE libNeurone)
target_compile_features(test_mnist PRIVATE cxx_std_17)
set_target_properties(test_mnist PROPERTIES CXX_EXTENSIONS OFF)

add_executable(test_module ${TEST})
target_include_directories(test_module PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
target_link_libraries(test_module PRIVATE libNeurone)
target_compile_features(test_module PRIVATE cxx_std_17)
set_target_properties(test_module PROPERTIES CXX_EXTENSIONS OFF)


if(MSVC)
    target_compile_options(libNeurone PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(libNeurone PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
endif()