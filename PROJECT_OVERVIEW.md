# Project Overview

## What is this project?

This project is a comprehensive benchmarking and demonstration platform for evaluating Large Language Model (LLM) performance across different hardware accelerators: **CPU, GPU, and NPU** (Neural Processing Unit).

## Purpose

The project aims to:
- **Benchmark** LLM inference performance across different hardware platforms
- **Compare** performance metrics (latency, throughput, token generation speed) with different precision settings (INT8, FP16, FP32)
- **Demonstrate** heterogeneous inference by running models on multiple accelerators in parallel
- **Visualize** results through an interactive Streamlit dashboard
- **Provide** a desktop GUI for real-time inference testing

## Key Components

### 1. Benchmarking System
- Benchmarks GPT-2 Large model across CPU, GPU, and NPU
- Tests multiple precision formats: INT8, FP16, FP32
- Collects metrics: latency, tokens/sec, throughput
- Saves results to JSON for analysis

### 2. Streamlit Dashboard
- Interactive web-based visualization of benchmark results
- Side-by-side comparison charts for hardware and precision combinations
- Real-time performance metrics display

### 3. Desktop GUI Application
- Tkinter-based GUI for live heterogeneous inference
- Runs Phi-3 (or other models) on GPU and GPT-2 ONNX on NPU in parallel
- Real-time prompt input and output display
- Performance metrics: latency, tokens/sec, CPU usage
- Threading for responsive UI during inference

### 4. Model Conversion Tools
- Scripts to convert Hugging Face models to ONNX format
- Support for static input shapes for NPU compatibility
- Precision conversion (FP32 → FP16 → INT8)

## Hardware Supported

| Hardware | Purpose | Technology |
|----------|---------|------------|
| **CPU** | Baseline inference | ONNX Runtime / PyTorch |
| **GPU** | High-performance inference | CUDA (NVIDIA) |
| **NPU** | Low-power, edge inference | OpenVINO / DirectML |

## Model Used

- **Primary Model**: GPT-2 Large (774M parameters)
- **Alternative for GUI**: Phi-3 Mini (3.8B parameters) for GPU, GPT-2 ONNX for NPU

## Use Cases

1. **Performance Analysis**: Understand how different hardware accelerators perform for LLM inference
2. **Hardware Selection**: Make informed decisions about which accelerator to use for specific workloads
3. **Precision Trade-offs**: Evaluate the performance vs. accuracy trade-offs of INT8, FP16, and FP32
4. **Heterogeneous Computing**: Demonstrate running multiple models on different accelerators simultaneously
5. **Edge Deployment**: Test NPU performance for edge device deployment scenarios

## Getting Started

See the main [README.md](README.md) for detailed setup instructions, or [TECHNICAL_ARCHITECTURE.md](TECHNICAL_ARCHITECTURE.md) for implementation details.
