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

# Model configurations
MODEL_CONFIGS = {
    'gpt2-medium': {'name': 'gpt2-medium', 'params': '345M', 'size': '1.4GB'},
    'pythia-410m': {'name': 'EleutherAI/pythia-410m', 'params': '410M', 'size': '1.6GB'},
    'pythia-800m': {'name': 'EleutherAI/pythia-800m', 'params': '800M', 'size': '3.1GB'},
}

# Default model to download
DEFAULT_MODEL = 'gpt2-medium'
OUTPUT_DIR = Path(__file__).parent.parent / "models"

def create_directories():
    """Create necessary directories."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {OUTPUT_DIR}")

def download_model(model_key):
    """Download model and tokenizer."""
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config['name']
    logger.info(f"Downloading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Save tokenizer
    tokenizer_path = OUTPUT_DIR / f"{model_key}-tokenizer"
    tokenizer_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_path)
    logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    return model, tokenizer, model_key

def export_to_onnx_fp32(model, tokenizer, model_key):
    """Export model to ONNX in FP32 format."""
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config['name']
    logger.info(f"Exporting {model_name} to ONNX FP32...")
    
    output_path = OUTPUT_DIR / f"{model_key}-fp32"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if OPTIMUM_AVAILABLE:
        try:
            ort_model = ORTModelForCausalLM.from_pretrained(
                model_name,
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
    
    # Wrapper module to handle model forward with proper arguments
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, input_ids, attention_mask):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=None)
    
    wrapped_model = ModelWrapper(model)
    
    torch.onnx.export(
        wrapped_model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        str(output_path / "model.onnx"),
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

def export_to_onnx_fp16(model, tokenizer, model_key):
    """Export model to ONNX in FP16 format by converting FP32 weights."""
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config['name']
    logger.info(f"Exporting {model_name} to ONNX FP16 by converting FP32 weights...")
    
    output_path = OUTPUT_DIR / f"{model_key}-fp16"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First ensure FP32 model exists
    fp32_path = OUTPUT_DIR / f"{model_key}-fp32"
    if not fp32_path.exists():
        export_to_onnx_fp32(model, tokenizer, model_key)
    
    # Use onnx to convert FP32 to FP16
    try:
        import onnx
        from onnxconverter_common import float16
        
        onnx_model_path = fp32_path / "model.onnx"
        fp16_model_path = output_path / "model.onnx"
        
        # Load FP32 model
        model_fp32 = onnx.load(onnx_model_path)
        
        # Convert to FP16
        model_fp16 = float16.convert_float_to_float16(model_fp32)
        
        # Save FP16 model
        onnx.save(model_fp16, fp16_model_path)
        
        tokenizer.save_pretrained(output_path)
        logger.info(f"FP16 model saved to {output_path}")
        
    except ImportError:
        logger.warning("onnxconverter-common not available, trying manual conversion")
        # Fallback: try using onnxruntime tools
        try:
            from onnxruntime.tools import convert_float_to_float16
            convert_float_to_float16(str(onnx_model_path), str(fp16_model_path))
            tokenizer.save_pretrained(output_path)
            logger.info(f"FP16 model saved to {output_path}")
        except ImportError:
            logger.warning("FP16 conversion tools not available, skipping")
    except Exception as e:
        logger.warning(f"FP16 export failed: {e}")
        logger.info("Skipping FP16 export")

def export_to_onnx_int8(model, tokenizer, model_key):
    """Export model to ONNX with INT8 quantization."""
    model_config = MODEL_CONFIGS[model_key]
    model_name = model_config['name']
    logger.info(f"Exporting {model_name} to ONNX INT8...")
    
    output_path = OUTPUT_DIR / f"{model_key}-int8"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # First export FP32 model
    fp32_path = OUTPUT_DIR / f"{model_key}-fp32"
    if not fp32_path.exists():
        export_to_onnx_fp32(model, tokenizer, model_key)
    
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
    
    # Download and convert all configured models
    for model_key in MODEL_CONFIGS.keys():
        logger.info(f"Processing model: {model_key}")
        
        # Download model
        model, tokenizer, model_key = download_model(model_key)
        
        # Export to different precisions
        export_to_onnx_fp32(model, tokenizer, model_key)
        export_to_onnx_fp16(model, tokenizer, model_key)
        
        # INT8 quantization
        export_to_onnx_int8(model, tokenizer, model_key)
        
        logger.info(f"Completed processing for {model_key}")
    
    logger.info("All models download and conversion complete!")

if __name__ == "__main__":
    main()
