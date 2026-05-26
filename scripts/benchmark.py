"""
Benchmark script for running inference on CPU, GPU, and NPU with different precision settings.
"""

import time
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run benchmarks on different hardware with different precision settings."""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.tokenizer = None
        
        # Hardware configurations
        self.hardware_configs = {
            'cpu': {'provider': 'CPUExecutionProvider'},
            'gpu': {'provider': 'CUDAExecutionProvider', 'provider_options': {}},
            'npu': {'provider': 'DmlExecutionProvider', 'provider_options': {}}
        }
    
    def load_tokenizer(self, model_path: Path):
        """Load tokenizer for the model."""
        if AutoTokenizer is None:
            raise ImportError("transformers not installed")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def check_hardware_available(self, hardware: str) -> bool:
        """Check if hardware is available."""
        if hardware == 'cpu':
            return True
        
        if hardware == 'gpu':
            try:
                available_providers = ort.get_available_providers()
                return 'CUDAExecutionProvider' in available_providers
            except:
                return False
        
        if hardware == 'npu':
            try:
                available_providers = ort.get_available_providers()
                return 'DmlExecutionProvider' in available_providers
            except:
                return False
        
        return False
    
    def load_model(self, model_path: Path, hardware: str) -> Optional[ort.InferenceSession]:
        """Load ONNX model for specific hardware."""
        if ort is None:
            raise ImportError("onnxruntime not installed")
        
        config = self.hardware_configs[hardware]
        
        try:
            session = ort.InferenceSession(
                str(model_path / "model.onnx"),
                providers=[config['provider']],
                provider_options=config.get('provider_options', {})
            )
            return session
        except Exception as e:
            logger.error(f"Failed to load model for {hardware}: {e}")
            return None
    
    def prepare_input(self, prompt: str = "The quick brown fox jumps over the lazy dog.") -> Dict[str, np.ndarray]:
        """Prepare input for inference."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not loaded")
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Convert input_ids to int64 as ONNX model expects int64
        if 'input_ids' in inputs:
            inputs['input_ids'] = inputs['input_ids'].astype(np.int64)
        
        return inputs
    
    def run_inference(self, session: ort.InferenceSession, inputs: Dict[str, np.ndarray], 
                     max_tokens: int = 50) -> Dict[str, Any]:
        """Run inference and collect metrics."""
        input_names = {input.name for input in session.get_inputs()}
        
        # Prepare inputs based on what the model expects
        onnx_inputs = {}
        for key, value in inputs.items():
            if key in input_names:
                onnx_inputs[key] = value
        
        # Warm-up run
        try:
            session.run(None, onnx_inputs)
        except:
            pass
        
        # Measure memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Run inference and measure time
        start_time = time.time()
        
        try:
            outputs = session.run(None, onnx_inputs)
            end_time = time.time()
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return None
        
        # Measure memory after
        memory_after = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Calculate metrics
        latency_ms = (end_time - start_time) * 1000
        memory_used_mb = memory_after - memory_before
        
        # Calculate token generation rate (tokens/second)
        # Assuming the output is logits, we count the sequence length
        tokens_generated = max_tokens  # Simplified assumption
        tokens_per_second = tokens_generated / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        # Throughput (total tokens processed / time)
        input_tokens = len(inputs['input_ids'][0])
        total_tokens = input_tokens + tokens_generated
        throughput = total_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_used_mb': memory_used_mb,
            'tokens_per_second': tokens_per_second,
            'throughput': throughput,
            'input_tokens': input_tokens,
            'total_tokens': total_tokens
        }
    
    def benchmark_model(self, model_name: str, precision: str, hardware: str, 
                       num_runs: int = 10) -> Dict[str, Any]:
        """Run benchmark for a specific model, precision, and hardware combination."""
        logger.info(f"Benchmarking {model_name} ({precision}) on {hardware}")
        
        if not self.check_hardware_available(hardware):
            logger.warning(f"{hardware} not available, skipping")
            return None
        
        model_path = self.models_dir / model_name
        if not model_path.exists():
            logger.error(f"Model path not found: {model_path}")
            return None
        
        # Load tokenizer
        if self.tokenizer is None:
            self.load_tokenizer(model_path)
        
        # Load model
        session = self.load_model(model_path, hardware)
        if session is None:
            return None
        
        # Prepare input
        inputs = self.prepare_input()
        
        # Run multiple iterations
        results = []
        for i in range(num_runs):
            result = self.run_inference(session, inputs)
            if result:
                results.append(result)
            logger.info(f"Run {i+1}/{num_runs} completed")
        
        if not results:
            return None
        
        # Calculate statistics
        latencies = [r['latency_ms'] for r in results]
        tps = [r['tokens_per_second'] for r in results]
        throughputs = [r['throughput'] for r in results]
        memory_usages = [r['memory_used_mb'] for r in results]
        
        return {
            'model': model_name,
            'precision': precision,
            'hardware': hardware,
            'num_runs': num_runs,
            'latency_ms_mean': np.mean(latencies),
            'latency_ms_std': np.std(latencies),
            'latency_ms_min': np.min(latencies),
            'latency_ms_max': np.max(latencies),
            'tokens_per_second_mean': np.mean(tps),
            'tokens_per_second_std': np.std(tps),
            'throughput_mean': np.mean(throughputs),
            'throughput_std': np.std(throughputs),
            'memory_used_mb_mean': np.mean(memory_usages),
            'memory_used_mb_std': np.std(memory_usages)
        }
