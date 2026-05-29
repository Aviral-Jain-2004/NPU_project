import openvino as ov
import numpy as np

print("Loading ONNX model...")
core = ov.Core()
model = core.read_model("gpt2_static.onnx")

print("Compiling model for NPU...")
try:
    compiled = core.compile_model(model, "NPU")
    device = "NPU"
    print("GPT-2 running on NPU")
except Exception as e:
    print(f"NPU compilation failed: {e}")
    print("Falling back to CPU...")
    compiled = core.compile_model(model, "CPU")
    device = "CPU"
    print("GPT-2 running on CPU")

print("Creating dummy input with shape [1, 10]...")
input_data = np.random.randint(0, 50257, (1, 10))  # GPT-2 vocab size is 50257

print(f"Running inference on {device}...")
output = compiled([input_data])[0]

print(f"Output shape: {output.shape}")
