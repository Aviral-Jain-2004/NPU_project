"""
Desktop GUI for heterogeneous LLM inference: Phi-3 on GPU + GPT-2 ONNX on NPU.
"""

import time
import threading
import tkinter as tk
from tkinter import scrolledtext, ttk

import numpy as np
import psutil

GPU_MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"
ONNX_PATH = "gpt2_static.onnx"

# Lazily loaded model handles (loaded once, reused across runs).
_tokenizer = None
_gpu_model = None
_gpu_device = None
_compiled_npu = None
_npu_device = None


def load_gpu_model():
    """Load Phi-3 with float16 and move to CUDA if available."""
    global _tokenizer, _gpu_model, _gpu_device
    if _gpu_model is not None:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        _tokenizer = AutoTokenizer.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
        # Load config first and fix rope_scaling before loading model
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(GPU_MODEL_NAME, trust_remote_code=True)
        # Fix rope_scaling compatibility issue with newer transformers
        if hasattr(config, 'rope_scaling') and config.rope_scaling is not None:
            if 'type' not in config.rope_scaling:
                config.rope_scaling = None
        
        _gpu_model = AutoModelForCausalLM.from_pretrained(
            GPU_MODEL_NAME,
            config=config,
            dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation='eager',
            use_cache=False,  # Disable DynamicCache for compatibility
        )
        _gpu_device = "cuda" if torch.cuda.is_available() else "cpu"
        _gpu_model.to(_gpu_device)
        _gpu_model.eval()
    except Exception as e:
        import traceback
        print(f"Error loading GPU model: {e}")
        print(traceback.format_exc())
        raise


def load_npu_model():
    """Compile the static GPT-2 ONNX model for the NPU (CPU fallback)."""
    global _compiled_npu, _npu_device
    if _compiled_npu is not None:
        return
    import openvino as ov

    core = ov.Core()
    model = core.read_model(ONNX_PATH)
    try:
        _compiled_npu = core.compile_model(model, "NPU")
        _npu_device = "NPU"
    except Exception:
        _compiled_npu = core.compile_model(model, "CPU")
        _npu_device = "CPU"


def run_npu(result_holder):
    """Run a dummy NPU inference with input shape [1, 10]."""
    input_data = np.random.randint(0, 50257, (1, 10))
    output = _compiled_npu([input_data])[0]
    result_holder["shape"] = output.shape


class App:
    def __init__(self, root):
        self.root = root
        root.title("Heterogeneous LLM Inference")
        root.geometry("700x650")

        tk.Label(root, text="Prompt:", anchor="w").pack(fill="x", padx=10, pady=(10, 0))
        self.prompt = scrolledtext.ScrolledText(root, height=6, wrap="word")
        self.prompt.pack(fill="both", expand=False, padx=10, pady=5)
        self.prompt.insert("1.0", "Explain what an NPU is in one sentence.")

        self.run_button = tk.Button(root, text="Run Inference", command=self.on_run)
        self.run_button.pack(pady=5)

        tk.Label(root, text="Output:", anchor="w").pack(fill="x", padx=10)
        self.output = scrolledtext.ScrolledText(root, height=12, wrap="word")
        self.output.pack(fill="both", expand=True, padx=10, pady=5)

        ttk.Separator(root, orient="horizontal").pack(fill="x", padx=10, pady=5)

        tk.Label(root, text="Metrics:", anchor="w", font=("", 10, "bold")).pack(fill="x", padx=10)
        self.latency_var = tk.StringVar(value="Latency: -")
        self.tps_var = tk.StringVar(value="Tokens/sec: -")
        self.cpu_var = tk.StringVar(value="CPU usage: -")
        self.status_var = tk.StringVar(value="GPU/NPU status: -")
        for var in (self.latency_var, self.tps_var, self.cpu_var, self.status_var):
            tk.Label(root, textvariable=var, anchor="w").pack(fill="x", padx=20)

    def on_run(self):
        self.run_button.config(state="disabled")
        self.set_output("Running inference...")
        prompt_text = self.prompt.get("1.0", "end").strip()
        threading.Thread(target=self.run_inference, args=(prompt_text,), daemon=True).start()

    def run_inference(self, prompt_text):
        try:
            import torch

            load_gpu_model()
            load_npu_model()

            npu_result = {}
            npu_thread = threading.Thread(target=run_npu, args=(npu_result,))

            psutil.cpu_percent(interval=None)  # prime CPU measurement
            npu_thread.start()

            inputs = _tokenizer(prompt_text, return_tensors="pt").to(_gpu_device)
            input_len = inputs["input_ids"].shape[1]

            start = time.time()
            with torch.no_grad():
                outputs = _gpu_model.generate(**inputs, max_new_tokens=50)
            latency = time.time() - start
            cpu_usage = psutil.cpu_percent(interval=None)

            npu_thread.join()

            output_ids = outputs[0][input_len:]
            text = _tokenizer.decode(output_ids, skip_special_tokens=True)
            tokens_generated = int(output_ids.shape[0])
            tps = tokens_generated / latency if latency > 0 else 0.0

            metrics = {
                "latency": latency,
                "tps": tps,
                "cpu": cpu_usage,
                "status": f"GPU: {_gpu_device} | NPU: {_npu_device} (out {npu_result.get('shape')})",
            }
            self.root.after(0, self.show_results, text, metrics)
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def set_output(self, text):
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)

    def show_results(self, text, metrics):
        self.set_output(text)
        self.latency_var.set(f"Latency: {metrics['latency']:.3f} s")
        self.tps_var.set(f"Tokens/sec: {metrics['tps']:.2f}")
        self.cpu_var.set(f"CPU usage: {metrics['cpu']:.1f} %")
        self.status_var.set(f"GPU/NPU status: {metrics['status']}")
        self.run_button.config(state="normal")

    def show_error(self, message):
        import traceback
        full_error = f"Error: {message}\n\n{traceback.format_exc()}"
        self.set_output(full_error)
        print(full_error)  # Also print to terminal
        self.run_button.config(state="normal")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
