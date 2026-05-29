import openvino as ov
import numpy as np
import torch
import torch.nn as nn

print("Creating dummy model...")

# Create a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.linear(x)

model_pytorch = SimpleModel()
model_pytorch.eval()

# Convert to OpenVINO model
print("Converting to OpenVINO model...")
model = ov.convert_model(model_pytorch, example_input=torch.rand(1, 10))

# Compile model for NPU
print("Compiling model for NPU...")
core = ov.Core()
compiled_model = core.compile_model(model, "NPU")

# Run inference with random input
print("Running inference...")
input_data = np.random.rand(1, 10)
result = compiled_model([input_data])[0]

print("NPU inference successful")
print(f"Output shape: {result.shape}")
