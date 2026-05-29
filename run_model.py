import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import threading
import numpy

def simulate_npu_task():
    """Simulate NPU workload with dummy computation."""
    start_time = time.time()
    duration = 2.0  # Run for ~2 seconds
    
    while time.time() - start_time < duration:
        # Dummy computation: small matrix multiplication
        matrix_a = numpy.random.rand(100, 100)
        matrix_b = numpy.random.rand(100, 100)
        result = numpy.dot(matrix_a, matrix_b)
        time.sleep(0.01)  # Simulate hardware execution

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

prompt = "Explain heterogeneous computing in simple terms."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

cpu_before = psutil.cpu_percent(interval=None)

# Start NPU workload in parallel
npu_thread = threading.Thread(target=simulate_npu_task)
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

# Wait for NPU task to complete
npu_thread.join()
print("NPU workload completed")

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
