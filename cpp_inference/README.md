# HybridArena C++ Inference

This directory contains the C++ runtime skeleton for low-latency MiniMOBA policy inference.

Default Python validation uses ONNX Runtime CPU. The C++ ONNX Runtime demo is optional and only builds when `ONNXRUNTIME_ROOT` points to an ONNX Runtime C++ install that provides `include/` and `lib/`.

TensorRT is intentionally not a default requirement. CUDA/TensorRT availability must be recorded as an environment capability before claiming acceleration results.
