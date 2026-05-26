"""
Data collection module for storing benchmark results.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd


class DataCollector:
    """Collect and store benchmark results."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = self.data_dir / "benchmark_results.json"
        self.results = self._load_existing_results()
    
    def _load_existing_results(self) -> List[Dict[str, Any]]:
        """Load existing results from file."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def add_result(self, result: Dict[str, Any]):
        """Add a benchmark result."""
        result['timestamp'] = datetime.now().isoformat()
        self.results.append(result)
        self._save_results()
    
    def _save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def get_filtered_results(self, model: str = None, precision: str = None, 
                           hardware: str = None) -> pd.DataFrame:
        """Filter results by criteria."""
        df = self.get_results_dataframe()
        
        if model:
            df = df[df['model'] == model]
        if precision:
            df = df[df['precision'] == precision]
        if hardware:
            df = df[df['hardware'] == hardware]
        
        return df
    
    def clear_results(self):
        """Clear all results."""
        self.results = []
        self._save_results()
