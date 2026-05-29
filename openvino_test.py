import openvino as ov
import numpy as np
import torch
import torch.nn as nn

print("Creating dummy model...")

# Create a simple PyTorch model with fixed input shape
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

model_pytorch = SimpleModel()
model_pytorch.eval()

# Convert to OpenVINO model with fixed input shape
print("Converting to OpenVINO model...")
dummy_input = torch.rand(1, 10)
model = ov.convert_model(model_pytorch, example_input=dummy_input)

# Reshape model to enforce static input dimensions
print("Enforcing static input shapes for NPU compatibility...")
model.reshape({0: [1, 10]})

# Compile model for NPU
print("Compiling model for NPU...")
core = ov.Core()
try:
    compiled_model = core.compile_model(model, "NPU")
    device = "NPU"
    print("NPU inference successful")
except Exception as e:
    print(f"NPU compilation failed: {e}")
    print("Falling back to CPU...")
    compiled_model = core.compile_model(model, "CPU")
    device = "CPU"

# Run inference with random input
print(f"Running inference on {device}...")
input_data = np.random.rand(1, 10)
result = compiled_model([input_data])[0]

print(f"Output shape: {result.shape}")
