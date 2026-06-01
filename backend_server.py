"""
Backend Flask server for GPU + NPU hybrid inference.
Loads models once at startup and exposes a /infer endpoint.

Run: python backend_server.py
"""

import time
import threading
from pathlib import Path

import numpy as np
import psutil
import torch
import openvino as ov
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPU_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ONNX_PATH = str(Path(__file__).resolve().parent / "gpt2_static.onnx")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load GPU model ONCE at startup
# ---------------------------------------------------------------------------
print(f"Loading tokenizer from {GPU_MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Fix rope_scaling compatibility issue with newer transformers
config = AutoConfig.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
    if "type" not in config.rope_scaling:
        config.rope_scaling = None

print(f"Loading model from {GPU_MODEL_NAME}...")
gpu_model = AutoModelForCausalLM.from_pretrained(
    GPU_MODEL_NAME,
    config=config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager",
)
gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_model.to(gpu_device)
gpu_model.eval()
print(f"GPU model loaded on: {gpu_device}")

# ---------------------------------------------------------------------------
# Load NPU model ONCE at startup
# ---------------------------------------------------------------------------
print("Loading ONNX model for NPU...")
core = ov.Core()
onnx_model = core.read_model(ONNX_PATH)
try:
    compiled_npu = core.compile_model(onnx_model, "NPU")
    npu_device = "NPU"
except Exception:
    compiled_npu = core.compile_model(onnx_model, "CPU")
    npu_device = "CPU"
print(f"NPU model compiled on: {npu_device}")


# ---------------------------------------------------------------------------
# NPU dummy inference (runs in a separate thread)
# ---------------------------------------------------------------------------
def run_npu(result_holder: dict):
    """Run a dummy NPU inference with input shape [1, 10]."""
    input_data = np.random.randint(0, 50257, (1, 10))
    output = compiled_npu([input_data])[0]
    result_holder["shape"] = str(output.shape)


# ---------------------------------------------------------------------------
# Core inference function
# ---------------------------------------------------------------------------
def run_inference(prompt: str) -> dict:
    """Run hybrid GPU + NPU inference and return results."""
    # --- Start NPU thread ---
    npu_result = {}
    npu_thread = threading.Thread(target=run_npu, args=(npu_result,))

    psutil.cpu_percent(interval=None)  # prime CPU measurement
    npu_thread.start()

    # --- GPU inference ---
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(gpu_device)

    start = time.time()
    with torch.no_grad():
        outputs = gpu_model.generate(
            inputs,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    latency = time.time() - start
    cpu_usage = psutil.cpu_percent(interval=None)

    # --- Wait for NPU ---
    npu_thread.join()

    # --- Decode only generated tokens ---
    generated_ids = outputs[0][inputs.shape[-1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_generated = int(generated_ids.shape[0])
    tokens_per_sec = tokens_generated / latency if latency > 0 else 0.0

    # --- GPU memory ---
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        gpu_mem = 0.0
        gpu_max = 0.0

    return {
        "output": generated_text,
        "latency": round(latency, 4),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "cpu_usage": round(cpu_usage, 1),
        "gpu_memory_mb": round(gpu_mem, 2),
        "gpu_peak_memory_mb": round(gpu_max, 2),
        "devices": f"GPU: {gpu_device} | NPU: {npu_device} (out {npu_result.get('shape')})",
    }


# ---------------------------------------------------------------------------
# Flask endpoint
# ---------------------------------------------------------------------------
@app.route("/infer", methods=["POST"])
def infer():
    data = request.json
    prompt = data.get("prompt", "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        result = run_inference(prompt)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gpu_device": gpu_device,
        "npu_device": npu_device,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
