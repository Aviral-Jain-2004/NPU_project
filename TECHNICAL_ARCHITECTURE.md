# Technical Architecture

## System Architecture

The project consists of three main components:

1. **Benchmarking Pipeline** - Scripts for running inference benchmarks
2. **Streamlit Dashboard** - Web-based visualization
3. **Desktop GUI** - Tkinter application for live inference

```
┌─────────────────┐
│   Benchmarking  │
│    Scripts      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Benchmark Data │
│   (JSON)        │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌────────┐
│ Streamlit│ │ Desktop│
│ Dashboard│ │   GUI  │
└────────┘ └────────┘
```

## 1. Benchmarking Pipeline

### Model Conversion (ONNX)

**File**: `scripts/download_model.py` / `export_gpt2_static_onnx.py`

The project uses ONNX (Open Neural Network Exchange) format for cross-platform compatibility:

```python
# Conversion process
model = GPT2LMHeadModel.from_pretrained("gpt2-large")
onnx_model = convert_to_onnx(model)
onnx_model.export("gpt2_static.onnx", input_shape=[1, 10])
```

**Why ONNX?**
- Hardware-agnostic intermediate representation
- Optimized by different runtimes (ONNX Runtime, OpenVINO)
- Supports precision quantization (FP32 → FP16 → INT8)
- Required for NPU compatibility

### Precision Formats

| Precision | Bit Width | Memory | Speed | Accuracy |
|-----------|-----------|--------|-------|----------|
| **FP32** | 32-bit | High | Baseline | Highest |
| **FP16** | 16-bit | 2x less | ~2x faster | Minimal loss |
| **INT8** | 8-bit | 4x less | ~4x faster | Noticeable loss |

### Inference Engines

#### CPU Inference
- **Engine**: ONNX Runtime or PyTorch
- **Execution**: Sequential on CPU cores
- **Use case**: Baseline measurement, no specialized hardware

#### GPU Inference
- **Engine**: PyTorch with CUDA backend
- **Execution**: Parallel on GPU cores
- **Key technologies**:
  - CUDA kernels for parallel computation
  - Tensor cores for FP16 acceleration
  - Memory bandwidth optimization
- **Requirements**: NVIDIA GPU with CUDA support

#### NPU Inference
- **Engine**: OpenVINO (Intel) or DirectML
- **Execution**: Hardware-accelerated matrix operations
- **Key technologies**:
  - OpenVINO Runtime for model compilation
  - NPU-specific instruction sets
  - Static input shapes (required for NPU)
- **Fallback**: CPU if NPU unavailable

### Benchmark Script Flow

```python
# scripts/benchmark.py pseudo-code
def benchmark(model_path, device, precision):
    # Load model
    model = load_onnx_model(model_path)
    
    # Compile for target device
    if device == "NPU":
        compiled = openvino.compile(model, "NPU")
    elif device == "GPU":
        compiled = onnxruntime.compile(model, "CUDA")
    else:
        compiled = onnxruntime.compile(model, "CPU")
    
    # Run inference
    start = time.time()
    output = compiled.run(input_data)
    latency = time.time() - start
    
    # Calculate metrics
    tokens = output.shape[1]
    tps = tokens / latency
    
    return {"latency": latency, "tokens_per_sec": tps}
```

## 2. Streamlit Dashboard

### Technology Stack
- **Framework**: Streamlit (Python web framework)
- **Visualization**: Plotly charts
- **Data**: JSON-based benchmark results

### Dashboard Components

```python
# dashboard/app.py structure
def main():
    st.set_page_config(title="NPU Benchmark Dashboard")
    
    # Load benchmark data
    data = load_benchmark_results("data/benchmark_results.json")
    
    # Create tabs for different precision levels
    tab1, tab2, tab3 = st.tabs(["INT8", "FP16", "FP32"])
    
    with tab1:
        # Display INT8 comparison charts
        plot_comparison(data["INT8"])
    
    # ... similar for FP16 and FP32
```

### Data Structure

```json
{
  "CPU": {
    "INT8": {"latency": 150, "tokens_per_sec": 20},
    "FP16": {"latency": 200, "tokens_per_sec": 15},
    "FP32": {"latency": 300, "tokens_per_sec": 10}
  },
  "GPU": {
    "INT8": {"latency": 50, "tokens_per_sec": 60},
    "FP16": {"latency": 30, "tokens_per_sec": 100},
    "FP32": {"latency": 40, "tokens_per_sec": 75}
  },
  "NPU": {
    "INT8": {"latency": 80, "tokens_per_sec": 40},
    "FP16": {"latency": 100, "tokens_per_sec": 30},
    "FP32": {"latency": 120, "tokens_per_sec": 25}
  }
}
```

## 3. Desktop GUI Application

### Technology Stack
- **UI Framework**: Tkinter (Python standard library)
- **Threading**: Python `threading` module for parallel execution
- **GPU Model**: Hugging Face Transformers (PyTorch)
- **NPU Model**: OpenVINO Runtime

### Architecture

```
┌─────────────────────────────────┐
│         Tkinter GUI             │
│  - Prompt Input                 │
│  - Run Button                   │
│  - Output Display               │
│  - Metrics Labels               │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│     Inference Orchestrator      │
│  (Main Thread)                  │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
┌──────────────┐ ┌──────────────┐
│ GPU Thread   │ │ NPU Thread   │
│ - Phi-3      │ │ - GPT-2 ONNX │
│ - PyTorch    │ │ - OpenVINO   │
└──────────────┘ └──────────────┘
```

### Key Implementation Details

#### Model Loading (Lazy)

```python
# gui_app.py - Lazy loading
_tokenizer = None
_gpu_model = None

def load_gpu_model():
    global _tokenizer, _gpu_model
    if _gpu_model is not None:
        return  # Already loaded
    
    _tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
    _gpu_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct")
    _gpu_model.to("cuda")
```

**Why lazy loading?**
- Faster GUI startup
- Only load models when user clicks "Run"
- Saves memory if GUI is opened but not used

#### Parallel Execution

```python
def run_inference(prompt_text):
    # Start NPU inference in background thread
    npu_result = {}
    npu_thread = threading.Thread(target=run_npu, args=(npu_result,))
    npu_thread.start()
    
    # Run GPU inference on main thread
    gpu_output = gpu_model.generate(prompt)
    
    # Wait for NPU to complete
    npu_thread.join()
    
    # Combine results
    return {"gpu": gpu_output, "npu": npu_result}
```

**Why threading?**
- Keeps GUI responsive during inference
- Enables true parallel execution across accelerators
- Prevents UI freeze

#### Prompt Formatting (Chat Template)

```python
# Phi-3 requires chat-formatted prompts
messages = [{"role": "user", "content": prompt_text}]
inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
)
```

**Why chat template?**
- Phi-3 is an instruction-tuned model
- Requires `<|user|>...<|end|><|assistant|>` format
- `apply_chat_template` handles this automatically

#### Generation Parameters

```python
outputs = model.generate(
    inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False
)
```

**Parameter explanations:**
- `do_sample=True`: Enables sampling (vs greedy)
- `temperature=0.7`: Controls randomness (lower = more deterministic)
- `top_p=0.9`: Nucleus sampling (keeps top 90% probability mass)
- `repetition_penalty=1.1`: Reduces repetitive output
- `pad_token_id`: Handles padding for batch processing
- `use_cache=False`: Disables KV cache (compatibility fix)

#### Output Decoding

```python
# Only decode newly generated tokens (not input)
output_ids = outputs[0][inputs.shape[-1]:]
text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
```

**Why slice from `inputs.shape[-1]`?**
- `outputs` contains: [input_tokens + generated_tokens]
- We only want the generated part
- `inputs.shape[-1]` = length of input tokens

### Metrics Collection

```python
# CPU usage
cpu_usage = psutil.cpu_percent(interval=None)

# Latency
start = time.time()
output = model.generate(...)
latency = time.time() - start

# Tokens per second
tokens_generated = len(output_ids)
tps = tokens_generated / latency
```

## Hardware-Specific Optimizations

### GPU Optimizations
- **Half-precision (FP16)**: Reduces memory usage, increases speed
- **Tensor cores**: Specialized hardware for matrix multiplication
- **CUDA graphs**: Reduces kernel launch overhead
- **Memory pinning**: Faster CPU-GPU data transfer

### NPU Optimizations
- **Static shapes**: Required for NPU compilation
- **INT8 quantization**: Leverages NPU's integer arithmetic units
- **Batch processing**: Maximizes NPU throughput
- **Layer fusion**: Reduces memory bandwidth

### CPU Optimizations
- **Multi-threading**: Parallel execution across cores
- **Vectorization**: SIMD instructions (AVX/AVX2)
- **Memory pooling**: Reduces allocation overhead

## Dependencies

### Core Libraries
- `torch`: PyTorch for GPU model loading and inference
- `transformers`: Hugging Face model zoo and tokenizers
- `openvino`: Intel OpenVINO runtime for NPU
- `onnxruntime`: ONNX model execution
- `streamlit`: Web dashboard framework
- `psutil`: System resource monitoring

### Hardware-Specific
- `nvidia-cuda-runtime`: CUDA support (GPU)
- `onnxruntime-directml`: DirectML support (NPU on Windows)

## File Structure

```
NPU_project/
├── PROJECT_OVERVIEW.md          # This file
├── TECHNICAL_ARCHITECTURE.md    # This file
├── README.md                    # Setup instructions
├── requirements.txt             # Python dependencies
├── gui_app.py                   # Desktop GUI
├── check_gpu.py                 # GPU verification
├── run_model.py                 # Standalone inference
├── export_gpt2_static_onnx.py   # Model conversion
├── run_gpt2_npu.py              # NPU inference demo
├── openvino_test.py             # OpenVINO testing
├── dashboard/
│   ├── app.py                   # Streamlit dashboard
│   └── pages/
│       └── Live_Model.py        # Live inference page
├── scripts/
│   ├── download_model.py        # Model download/convert
│   ├── benchmark.py             # Benchmark execution
│   ├── data_collector.py        # Data collection
│   └── run_benchmarks.py        # Orchestration
├── data/
│   └── benchmark_results.json   # Benchmark data
└── models/
    └── *.onnx                   # ONNX model files
```

## Future Enhancements

1. **Model Support**: Add more models (Llama, Mistral, etc.)
2. **Precision Modes**: Add 4-bit quantization (GPTQ, AWQ)
3. **Batch Inference**: Support for batch processing
4. **Real-time Streaming**: Token-by-token output streaming
5. **Multi-GPU**: Support for multi-GPU setups
6. **Cloud Integration**: Support for cloud-based inference
7. **MLflow Integration**: Experiment tracking and versioning
