# NPU Project - LLM Benchmarking Dashboard

This project benchmarks LLM models (GPT-2 Large) on CPU, GPU, and NPU with different precision settings (int8, fp16, fp32) and visualizes the results in a dashboard.

## Project Structure

```
NPU_project/
├── models/              # ONNX model files
├── data/               # Benchmark results
├── scripts/
│   ├── download_model.py      # Download/convert models to ONNX
│   ├── benchmark.py           # Run inference benchmarks
│   ├── data_collector.py      # Collect and store metrics
│   └── run_benchmarks.py      # Main orchestration script
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── check_gpu.py               # GPU verification script
├── run_model.py               # Standalone model inference script
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** You may need to install different onnxruntime packages depending on your hardware:
- CPU only: `pip install onnxruntime`
- GPU (CUDA): `pip install onnxruntime-gpu`
- NPU (DirectML): `pip install onnxruntime-directml`

**GPU Support:** For GPU acceleration with CUDA 12.8 support, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### 2. Download Models

Run the model download script to get GPT-2 Large in ONNX format with different precisions:

```bash
python scripts/download_model.py
```

This will download and convert GPT-2 Large to ONNX format with:
- FP32 precision
- FP16 precision
- INT8 quantization

### 3. Run Benchmarks

Run the benchmark script on your remote desktop (where you have the hardware):

```bash
python scripts/run_benchmarks.py
```

This will run inference on:
- CPU
- GPU (if available)
- NPU (if available)

With precision settings:
- INT8
- FP16
- FP32

### 4. View Dashboard

After benchmarks complete, start the Streamlit dashboard:

```bash
streamlit run dashboard/app.py
```

## Metrics Collected

For each combination of hardware and precision, the dashboard shows:
- **Latency**: Time per inference (ms)
- **Token Generation**: Tokens generated per second
- **Throughput**: Total tokens processed per second

## Comparison Structure

The dashboard displays 9 comparison charts:

### INT8 Precision
- Latency: CPU vs GPU vs NPU
- Token Generation: CPU vs GPU vs NPU
- Throughput: CPU vs GPU vs NPU

### FP16 Precision
- Latency: CPU vs GPU vs NPU
- Token Generation: CPU vs GPU vs NPU
- Throughput: CPU vs GPU vs NPU

### FP32 Precision
- Latency: CPU vs GPU vs NPU
- Token Generation: CPU vs GPU vs NPU
- Throughput: CPU vs GPU vs NPU

## Notes

- Ensure you have the appropriate hardware (GPU with CUDA, NPU with DirectML support) before running benchmarks
- The benchmark script will skip hardware that is not available
- Results are saved to `data/benchmark_results.json`
