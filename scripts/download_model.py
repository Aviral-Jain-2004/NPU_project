"""
Download and convert GPT-2 Large to ONNX format with different precision settings.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from optimum.exporters.onnx import OnnxConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "gpt2-large"
OUTPUT_DIR = Path(__file__).parent.parent / "models"

def create_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

def download_model():
    """Download GPT-2 Large model and tokenizer."""
    logger.info(f"Downloading {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    
    # Save tokenizer
    tokenizer_path = OUTPUT_DIR / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    return model, tokenizer

def export_to_onnx_fp32(model, tokenizer):
    """Export model to ONNX in FP32 format."""
    logger.info("Exporting to ONNX FP32...")
    
    output_path = OUTPUT_DIR / "gpt2-large-fp32"
    
    ort_model = ORTModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        provider="CPUExecutionProvider",
    )
    ort_model.save_pretrained(output_path)
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"FP32 model saved to {output_path}")

def export_to_onnx_fp16(model, tokenizer):
    """Export model to ONNX in FP16 format."""
    logger.info("Exporting to ONNX FP16...")
    
    output_path = OUTPUT_DIR / "gpt2-large-fp16"
    
    ort_model = ORTModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        provider="CPUExecutionProvider",
        float16=True,
    )
    ort_model.save_pretrained(output_path)
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"FP16 model saved to {output_path}")

def export_to_onnx_int8(model, tokenizer):
    """Export model to ONNX with INT8 quantization."""
    logger.info("Exporting to ONNX INT8...")
    
    output_path = OUTPUT_DIR / "gpt2-large-int8"
    
    # First export to ONNX
    ort_model = ORTModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        provider="CPUExecutionProvider",
    )
    
    # Apply dynamic quantization
    from optimum.onnxruntime.configuration import QuantizationConfig
    from optimum.onnxruntime import ORTQuantizer
    
    quantization_config = QuantizationConfig(
        is_static=False,
        format="int8",
        per_channel=False,
        reduce_range=False,
    )
    
    quantizer = ORTQuantizer.from_pretrained(ort_model)
    quantizer.quantize(
        save_dir=output_path,
        quantization_config=quantization_config,
    )
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"INT8 model saved to {output_path}")

def main():
    """Main function to download and convert models."""
    logger.info("Starting model download and conversion...")
    
    create_directories()
    
    # Download model
    model, tokenizer = download_model()
    
    # Export to different precisions
    export_to_onnx_fp32(model, tokenizer)
    export_to_onnx_fp16(model, tokenizer)
    
    # INT8 quantization
    try:
        export_to_onnx_int8(model, tokenizer)
    except Exception as e:
        logger.warning(f"INT8 export failed: {e}")
        logger.info("You may need to manually quantize the model using optimum-cli")
    
    logger.info("Model download and conversion complete!")

if __name__ == "__main__":
    main()
