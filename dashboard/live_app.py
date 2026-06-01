"""
Live Hybrid LLM Inference (GPU + NPU) – Streamlit App
Runs on port 8502, independent of the benchmark dashboard.
"""

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np
import psutil
import streamlit as st
import torch
import openvino as ov
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GPU_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
ONNX_PATH = str(Path(__file__).resolve().parent.parent / "gpt2_static.onnx")

# ---------------------------------------------------------------------------
# Cached model loaders (loaded once per Streamlit process)
# ---------------------------------------------------------------------------

@st.cache_resource
def load_gpu_model():
    """Load Phi-3 with float16 and move to CUDA."""
    tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Fix rope_scaling compatibility issue with newer transformers
    config = AutoConfig.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "type" not in config.rope_scaling:
            config.rope_scaling = None

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
    return tokenizer, gpu_model, gpu_device


@st.cache_resource
def load_npu_model():
    """Compile the static GPT-2 ONNX model for the NPU (CPU fallback)."""
    core = ov.Core()
    model = core.read_model(ONNX_PATH)
    try:
        compiled_npu = core.compile_model(model, "NPU")
        npu_device = "NPU"
    except Exception:
        compiled_npu = core.compile_model(model, "CPU")
        npu_device = "CPU"
    return compiled_npu, npu_device


def run_npu(compiled_npu, result_holder: dict):
    """Run a dummy NPU inference with input shape [1, 10]."""
    input_data = np.random.randint(0, 50257, (1, 10))
    output = compiled_npu([input_data])[0]
    result_holder["shape"] = output.shape


def run_inference(prompt_text: str) -> dict:
    """Run hybrid GPU + NPU inference and return results with metrics."""
    tokenizer, gpu_model, gpu_device = load_gpu_model()
    compiled_npu, npu_device = load_npu_model()

    # --- Start NPU thread ---
    npu_result = {}
    npu_thread = threading.Thread(target=run_npu, args=(compiled_npu, npu_result))

    psutil.cpu_percent(interval=None)  # prime CPU measurement
    npu_thread.start()

    # --- GPU inference ---
    messages = [{"role": "user", "content": prompt_text}]
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
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tokens_generated = int(generated_ids.shape[0])
    tps = tokens_generated / latency if latency > 0 else 0.0

    # --- GPU memory ---
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / (1024 ** 2)
        gpu_max = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        gpu_mem = 0.0
        gpu_max = 0.0

    return {
        "text": text,
        "latency": latency,
        "tokens": tokens_generated,
        "tps": tps,
        "cpu": cpu_usage,
        "gpu_mem": gpu_mem,
        "gpu_max": gpu_max,
        "status": f"GPU: {gpu_device} | NPU: {npu_device} (out {npu_result.get('shape')})",
    }


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Live Hybrid LLM Inference", layout="wide")
st.title("Live Hybrid LLM Inference (GPU + NPU)")
st.caption(
    "Phi-3 Mini on GPU  ·  GPT-2 ONNX on NPU  ·  Parallel execution"
)

# --- Check ONNX file ---
if not os.path.exists(ONNX_PATH):
    st.error(
        f"ONNX model not found at `{ONNX_PATH}`. "
        "Run `python export_gpt2_static_onnx.py` first."
    )
    st.stop()

# --- Input ---
prompt = st.text_input("Enter your prompt:", value="Explain heterogeneous computing in simple terms.")

if st.button("Run Inference"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Loading models & running inference …"):
            try:
                result = run_inference(prompt.strip())

                # --- Output ---
                st.subheader("Generated Text")
                st.text_area("Output", value=result["text"], height=200)

                # --- Metrics ---
                st.subheader("Performance Metrics")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Latency", f"{result['latency']:.2f} s")
                col2.metric("Tokens / sec", f"{result['tps']:.2f}")
                col3.metric("CPU Usage", f"{result['cpu']:.1f} %")
                col4.metric("Tokens Generated", str(result["tokens"]))

                st.subheader("Device Info")
                col5, col6, col7 = st.columns(3)
                col5.metric("GPU Memory", f"{result['gpu_mem']:.0f} MB")
                col6.metric("GPU Peak Memory", f"{result['gpu_max']:.0f} MB")
                col7.metric("Devices", result["status"])

            except Exception as e:
                st.error(f"Inference failed: {e}")
                import traceback
                st.code(traceback.format_exc())
