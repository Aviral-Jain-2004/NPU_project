import openvino as ov
import numpy as np

print("Creating dummy model...")

# Create a simple function that multiplies input by random weights
def simple_model(input_data):
    weights = np.random.rand(10, 10)
    return np.dot(input_data, weights)

# Convert to OpenVINO model
print("Converting to OpenVINO model...")
model = ov.convert_model(simple_model, example_input=np.random.rand(1, 10))

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
