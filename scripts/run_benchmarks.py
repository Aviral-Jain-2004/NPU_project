"""
Main orchestration script to run all benchmarks.
"""

import sys
from pathlib import Path
import logging
from tqdm import tqdm

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from benchmark import BenchmarkRunner
from data_collector import DataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function to run all benchmarks."""
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    
    logger.info(f"Project root: {project_root}")
    logger.info(f"Models directory: {models_dir}")
    logger.info(f"Data directory: {data_dir}")
    
    # Initialize data collector
    collector = DataCollector(data_dir)
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(models_dir)
    
    # Define benchmark configurations
    # Model names should match the directories created by download_model.py
    model_configs = [
        ("gpt2-fp32", "fp32"),
        ("gpt2-fp16", "fp16"),
        ("gpt2-int8", "int8"),
    ]
    
    # Hardware to test
    hardware_list = ["cpu", "gpu", "npu"]
    
    # Number of runs per configuration
    num_runs = 10
    
    logger.info("Starting benchmark suite...")
    logger.info(f"Testing {len(model_configs)} model configurations")
    logger.info(f"Testing hardware: {hardware_list}")
    logger.info(f"Runs per configuration: {num_runs}")
    
    # Run benchmarks
    total_configs = len(model_configs) * len(hardware_list)
    progress_bar = tqdm(total=total_configs, desc="Running benchmarks")
    
    for model_name, precision in model_configs:
        for hardware in hardware_list:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {model_name} ({precision}) on {hardware}")
            logger.info(f"{'='*60}")
            
            try:
                result = runner.benchmark_model(
                    model_name=model_name,
                    precision=precision,
                    hardware=hardware,
                    num_runs=num_runs
                )
                
                if result:
                    collector.add_result(result)
                    logger.info(f"✓ Benchmark completed successfully")
                    logger.info(f"  Latency: {result['latency_ms_mean']:.2f} ms")
                    logger.info(f"  Token Generation: {result['tokens_per_second_mean']:.2f} tokens/s")
                    logger.info(f"  Throughput: {result['throughput_mean']:.2f} tokens/s")
                else:
                    logger.warning(f"✗ Benchmark failed or skipped")
            
            except Exception as e:
                logger.error(f"Error during benchmark: {e}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    logger.info("\n" + "="*60)
    logger.info("Benchmark suite completed!")
    logger.info(f"Results saved to: {collector.results_file}")
    logger.info("="*60)
    
    # Print summary
    df = collector.get_results_dataframe()
    if not df.empty:
        logger.info("\nSummary of results:")
        logger.info(f"Total successful benchmarks: {len(df)}")
        
        logger.info("\nResults by hardware:")
        for hw in df['hardware'].unique():
            hw_results = df[df['hardware'] == hw]
            logger.info(f"  {hw.upper()}: {len(hw_results)} benchmarks")
        
        logger.info("\nResults by precision:")
        for prec in df['precision'].unique():
            prec_results = df[df['precision'] == prec]
            logger.info(f"  {prec.upper()}: {len(prec_results)} benchmarks")


if __name__ == "__main__":
    main()
