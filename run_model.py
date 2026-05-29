import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import threading
import openvino as ov
import numpy as np

def run_npu_model():
    """Run NPU model inference in parallel."""
    input_data = np.random.randint(0, 50257, (1, 10))  # GPT-2 vocab size
    compiled_model([input_data])
    print("NPU model completed")

model_name = "microsoft/Phi-3-mini-4k-instruct"

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print(f"Loading model from {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

# Load ONNX model for NPU
print("Loading ONNX model for NPU...")
core = ov.Core()
onnx_model = core.read_model("gpt2_static.onnx")
compiled_model = core.compile_model(onnx_model, "NPU")
print("Loaded and compiled ONNX model for NPU")

prompt = "Explain heterogeneous computing in simple terms."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

cpu_before = psutil.cpu_percent(interval=None)

# Start NPU model in parallel
npu_thread = threading.Thread(target=run_npu_model)
npu_thread.start()
print("NPU workload running in parallel...")

print("Running inference...")
start_time = time.time()
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7
    )
end_time = time.time()

# Wait for NPU model to complete
npu_thread.join()

latency = end_time - start_time
cpu_after = psutil.cpu_percent(interval=None)

if torch.cuda.is_available():
    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    gpu_max_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
else:
    gpu_memory_allocated = 0
    gpu_max_memory = 0

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
tokens_per_second = tokens_generated / latency

print("\n--- Results ---")
print(f"Generated Text:\n{generated_text}")

print("\n--- Performance ---")
print(f"Latency:          {latency:.4f} seconds")
print(f"Tokens Generated: {tokens_generated}")
print(f"Tokens/sec:       {tokens_per_second:.2f}")

print("\n--- System Usage ---")
print(f"CPU Usage Before: {cpu_before:.1f} %")
print(f"CPU Usage After:  {cpu_after:.1f} %")

if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
    print(f"GPU Max Memory Used:   {gpu_max_memory:.2f} MB")
else:
    print("GPU Memory Allocated: N/A")
    print("GPU Max Memory Used:   N/A")
