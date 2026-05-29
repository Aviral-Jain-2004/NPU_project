import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

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

latency = end_time - start_time
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
tokens_per_second = tokens_generated / latency

print("\n--- Results ---")
print(f"Generated Text:\n{generated_text}")
print(f"\nLatency:          {latency:.4f} seconds")
print(f"Tokens Generated: {tokens_generated}")
print(f"Tokens/Second:    {tokens_per_second:.2f}")
