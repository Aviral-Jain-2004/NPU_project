import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading gpt2 model...")
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

print("Creating dummy input with fixed shape [1, 10]...")
dummy_input = torch.randint(0, tokenizer.vocab_size, (1, 10))

print("Exporting to ONNX with static shapes...")
torch.onnx.export(
    model,
    dummy_input,
    "gpt2_static.onnx",
    input_names=["input_ids"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes=None  # No dynamic axes for NPU compatibility
)

print("Exported gpt2_static.onnx successfully")
