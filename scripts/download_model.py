"""
Download and convert GPT-2 Large to ONNX format with different precision settings.
"""

import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

try:
    from optimum.onnxruntime import ORTModelForCausalLM
    OPTIMUM_AVAILABLE = True
except ImportError:
    OPTIMUM_AVAILABLE = False
    logging.warning("optimum.onnxruntime not available, will use alternative export method")

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
    output_path.mkdir(parents=True, exist_ok=True)
    
    if OPTIMUM_AVAILABLE:
        try:
            ort_model = ORTModelForCausalLM.from_pretrained(
                MODEL_NAME,
                export=True,
                provider="CPUExecutionProvider",
            )
            ort_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(f"FP32 model saved to {output_path}")
            return
        except Exception as e:
            logger.warning(f"Optimum export failed: {e}, falling back to manual export")
    
    # Fallback: manual export using torch
    logger.info("Using manual ONNX export...")
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    
    # Wrapper function to handle model forward with proper arguments
    def model_wrapper(input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=None)
    
    torch.onnx.export(
        model_wrapper,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path / "model.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14
    )
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"FP32 model saved to {output_path}")

def export_to_onnx_fp16(model, tokenizer):
    """Export model to ONNX in FP16 format."""
    logger.info("Exporting to ONNX FP16...")
    
    output_path = OUTPUT_DIR / "gpt2-large-fp16"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if OPTIMUM_AVAILABLE:
        try:
            ort_model = ORTModelForCausalLM.from_pretrained(
                MODEL_NAME,
                export=True,
                provider="CPUExecutionProvider",
                float16=True,
            )
            ort_model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            logger.info(f"FP16 model saved to {output_path}")
            return
        except Exception as e:
            logger.warning(f"Optimum export failed: {e}, falling back to manual export")
    
    # Fallback: convert model to FP16 then export
    logger.info("Using manual ONNX export with FP16...")
    model_fp16 = model.half()
    dummy_input = tokenizer("Hello world", return_tensors="pt")
    
    # Wrapper function to handle model forward with proper arguments
    def model_wrapper_fp16(input_ids, attention_mask):
        return model_fp16(input_ids=input_ids, attention_mask=attention_mask, past_key_values=None)
    
    torch.onnx.export(
        model_wrapper_fp16,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path / "model.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=14
    )
    
    tokenizer.save_pretrained(output_path)
    logger.info(f"FP16 model saved to {output_path}")

def export_to_onnx_int8(model, tokenizer):
    """Export model to ONNX with INT8 quantization."""
    logger.info("Exporting to ONNX INT8...")
    
    output_path = OUTPUT_DIR / "gpt2-large-int8"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First export FP32 model
    fp32_path = OUTPUT_DIR / "gpt2-large-fp32"
    if not fp32_path.exists():
        export_to_onnx_fp32(model, tokenizer)
    
    # Use onnxruntime for quantization
    try:
        import onnx
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        onnx_model_path = fp32_path / "model.onnx"
        quantized_model_path = output_path / "model.onnx"
        
        quantize_dynamic(
            onnx_model_path,
            quantized_model_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True,
            per_channel=False,
            reduce_range=False
        )
        
        tokenizer.save_pretrained(output_path)
        logger.info(f"INT8 model saved to {output_path}")
        
    except ImportError:
        logger.warning("onnxruntime quantization not available")
        logger.info("Skipping INT8 export - you can manually quantize using onnxruntime tools")
    except Exception as e:
        logger.warning(f"INT8 export failed: {e}")
        logger.info("You may need to manually quantize the model using onnxruntime tools")

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
    export_to_onnx_int8(model, tokenizer)
    
    logger.info("Model download and conversion complete!")

if __name__ == "__main__":
    main()
