import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import threading

def run_auxiliary_model():
    """Run auxiliary model inference in parallel."""
    prompt = "Hello, how are you?"
    inputs = aux_tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = aux_model.generate(**inputs, max_new_tokens=10)

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

# Load auxiliary model (gpt2) on CPU
aux_model_name = "gpt2"
print(f"Loading auxiliary model from {aux_model_name}...")
aux_tokenizer = AutoTokenizer.from_pretrained(aux_model_name)
aux_model = AutoModelForCausalLM.from_pretrained(aux_model_name)
aux_model = aux_model.to("cpu")
print("Loaded auxiliary model (gpt2) on CPU")

prompt = "Explain heterogeneous computing in simple terms."

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

cpu_before = psutil.cpu_percent(interval=None)

# Start auxiliary model in parallel
aux_thread = threading.Thread(target=run_auxiliary_model)
aux_thread.start()
print("Auxiliary model running in parallel...")

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

# Wait for auxiliary model to complete
aux_thread.join()
print("Auxiliary model completed")

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
