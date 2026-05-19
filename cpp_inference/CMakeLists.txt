cmake_minimum_required(VERSION 3.20)
project(hybrid_arena_cpp_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(observation_codec src/observation_codec.cpp)
target_include_directories(observation_codec PUBLIC include)

add_executable(observation_codec_test tests/test_observation_codec.cpp)
target_link_libraries(observation_codec_test PRIVATE observation_codec)

if(DEFINED ENV{ONNXRUNTIME_ROOT})
  add_executable(hybrid_arena_infer src/main.cpp)
  target_include_directories(hybrid_arena_infer PRIVATE include "$ENV{ONNXRUNTIME_ROOT}/include")
  target_link_directories(hybrid_arena_infer PRIVATE "$ENV{ONNXRUNTIME_ROOT}/lib")
  target_link_libraries(hybrid_arena_infer PRIVATE observation_codec onnxruntime)
else()
  message(STATUS "ONNXRUNTIME_ROOT is not set; C++ ONNX Runtime demo target is skipped.")
endif()
