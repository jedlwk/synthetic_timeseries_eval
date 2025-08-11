"""
Time Series Data Loader Module

This module provides comprehensive data loading functionality for both conditional and 
unconditional time series datasets. It handles various file naming conventions and 
data structures commonly found in synthetic time series evaluation tasks.

Key Features:
- Supports both conditional (with static features) and unconditional time series
- Handles multiple synthetic variants (tsv1, tsv2, original_noise)
- Robust error handling and validation
- Variable-length time series support
- Automatic data structure detection and normalization

"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

class TimeSeriesDataLoader:
    """
    A comprehensive data loader for time series datasets supporting both conditional
    and unconditional generation evaluation tasks.
    
    This class handles the complex data loading requirements for synthetic time series
    evaluation, including multiple file naming conventions, missing files, and 
    variable-length sequences.
    
    Attributes:
        data_root (Path): Root directory containing the TimeSeries data
        conditional_path (Path): Path to conditional generation data
        unconditional_path (Path): Path to unconditional generation data
    
    Example:
        >>> loader = TimeSeriesDataLoader("TimeSeries")
        >>> static_data, ts_data = loader.load_conditional_data("tsv2")
        >>> print(f"Loaded {len(ts_data)} time series with {len(static_data)} conditions")
    """
    def __init__(self, data_root: str):
        """
        Initialize the TimeSeriesDataLoader with the root data directory.
        
        Args:
            data_root (str): Path to the root directory containing TimeSeries data.
                           Expected to contain 'conditional generation' and 
                           'unconditional generation' subdirectories.
        
        Raises:
            FileNotFoundError: If the data_root directory doesn't exist.
        """
        self.data_root = Path(data_root)
        self.conditional_path = self.data_root / "conditional generation"
        self.unconditional_path = self.data_root / "unconditional generation"
    
    def load_conditional_data(self, variant: str = "original") -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Load conditional time series data including static conditions and corresponding
        time series files.
        
        This method handles the complex mapping between static conditions (.id column)
        and their corresponding time series files. It supports multiple naming conventions
        across different synthetic variants.
        
        Args:
            variant (str): The data variant to load. Options include:
                         - "original": Ground truth data (series_XXX.csv format)
                         - "tsv1": Synthetic variant 1 (sample_X.csv format)  
                         - "tsv2": Synthetic variant 2 (X.csv format)
                         - "original_noise": Synthetic variant 3 (series_XXX.csv format)
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]: A tuple containing:
                - static_data: DataFrame with static conditions (.id, features)
                - time_series_data: Dictionary mapping series_id -> time series DataFrame
        
        Raises:
            FileNotFoundError: If required static.csv or time_series directory not found.
            
        Example:
            >>> loader = TimeSeriesDataLoader("TimeSeries")
            >>> static, ts_dict = loader.load_conditional_data("tsv2")
            >>> print(f"Static conditions shape: {static.shape}")
            >>> print(f"Number of time series: {len(ts_dict)}")
            >>> # Access specific time series by ID
            >>> series_0 = ts_dict['0']  # Time series for condition ID 0
        """
        variant_path = self.conditional_path / variant
        
        if variant == "original_noise":
            static_file = self.conditional_path / "original" / "static.csv"
            ts_folder = variant_path
        else:
            static_file = variant_path / "static.csv"
            ts_folder = variant_path / "time_series"
        
        if not static_file.exists():
            raise FileNotFoundError(f"Static file not found: {static_file}")
        
        static_data = pd.read_csv(static_file)
        
        if not ts_folder.exists():
            raise FileNotFoundError(f"Time series folder not found: {ts_folder}")
        
        time_series_data = {}
        
        for _, row in static_data.iterrows():
            series_id = str(int(row['.id']))
            
            if variant in ["original", "original_noise"]:
                ts_file = ts_folder / f"series_{series_id.zfill(3)}.csv"
            elif variant == "tsv1":
                ts_file = ts_folder / f"sample_{series_id}.csv"
            else:
                ts_file = ts_folder / f"{series_id}.csv"
            
            if ts_file.exists():
                ts_df = pd.read_csv(ts_file)
                time_series_data[series_id] = ts_df
            else:
                print(f"Warning: Time series file not found: {ts_file}")
        
        return static_data, time_series_data
    
    def load_unconditional_data(self, variant: str = "original") -> Dict[str, pd.DataFrame]:
        """
        Load unconditional time series data where each CSV file represents a standalone
        time series without associated static conditions.
        
        This method handles different directory structures for original vs synthetic
        unconditional data variants, automatically detecting and loading all CSV files
        in the appropriate directory.
        
        Args:
            variant (str): The data variant to load. Options include:
                         - "original": Ground truth unconditional data
                         - "tsv2": Synthetic variant (stored in ts/ subdirectory)
        
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping filename (without extension) 
                                   to time series DataFrame
        
        Raises:
            FileNotFoundError: If the data folder for the specified variant doesn't exist.
            
        Example:
            >>> loader = TimeSeriesDataLoader("TimeSeries")
            >>> ts_data = loader.load_unconditional_data("tsv2")
            >>> print(f"Loaded {len(ts_data)} unconditional time series")
            >>> # Access specific time series by filename
            >>> series = ts_data['0']  # Time series from 0.csv
        """
        variant_path = self.unconditional_path / variant
        
        if variant == "original":
            data_folder = variant_path
        else:
            data_folder = variant_path / "ts"
        
        if not data_folder.exists():
            raise FileNotFoundError(f"Data folder not found: {data_folder}")
        
        time_series_data = {}
        
        for ts_file in data_folder.glob("*.csv"):
            series_id = ts_file.stem
            ts_df = pd.read_csv(ts_file)
            time_series_data[series_id] = ts_df
        
        return time_series_data
    
    def get_available_variants(self, data_type: str = "conditional") -> List[str]:
        """
        Get list of available data variants for the specified data type.
        
        Args:
            data_type (str): Type of data variants to list ("conditional" or "unconditional")
        
        Returns:
            List[str]: Sorted list of available variant names
        
        Example:
            >>> loader = TimeSeriesDataLoader("TimeSeries")
            >>> variants = loader.get_available_variants("conditional")
            >>> print(variants)  # ['original', 'original_noise', 'tsv1', 'tsv2']
        """
        if data_type == "conditional":
            path = self.conditional_path
        else:
            path = self.unconditional_path
        
        variants = [d.name for d in path.iterdir() if d.is_dir()]
        return sorted(variants)
    
    def validate_data_structure(self) -> Dict[str, Dict[str, int]]:
        """
        Validate the structure and integrity of all available data variants.
        
        This method attempts to load each variant and reports statistics about
        the data structure, including number of files, data quality indicators,
        and any loading errors encountered.
        
        Returns:
            Dict[str, Dict[str, int]]: Nested dictionary containing validation results
                                     for each data type and variant
        
        Example:
            >>> loader = TimeSeriesDataLoader("TimeSeries")
            >>> results = loader.validate_data_structure()
            >>> print(results['conditional']['tsv2']['static_rows'])  # Number of conditions
        """
        validation_results = {
            "conditional": {},
            "unconditional": {}
        }
        
        for variant in self.get_available_variants("conditional"):
            try:
                static_data, ts_data = self.load_conditional_data(variant)
                validation_results["conditional"][variant] = {
                    "static_rows": len(static_data),
                    "time_series_files": len(ts_data),
                    "avg_ts_length": np.mean([len(df) for df in ts_data.values()]) if ts_data else 0
                }
            except Exception as e:
                validation_results["conditional"][variant] = {"error": str(e)}
        
        for variant in self.get_available_variants("unconditional"):
            try:
                ts_data = self.load_unconditional_data(variant)
                validation_results["unconditional"][variant] = {
                    "time_series_files": len(ts_data),
                    "avg_ts_length": np.mean([len(df) for df in ts_data.values()]) if ts_data else 0
                }
            except Exception as e:
                validation_results["unconditional"][variant] = {"error": str(e)}
        
        return validation_results
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive summary of all available data including variants and validation.
        
        Returns:
            Dict: Complete summary containing available variants and validation results
        
        Example:
            >>> loader = TimeSeriesDataLoader("TimeSeries")
            >>> summary = loader.get_data_summary()
            >>> print(f"Found {len(summary['conditional_variants'])} conditional variants")
        """
        summary = {
            "conditional_variants": self.get_available_variants("conditional"),
            "unconditional_variants": self.get_available_variants("unconditional"),
            "validation": self.validate_data_structure()
        }
        return summary

def normalize_time_series_length(ts_dict: Dict[str, pd.DataFrame], 
                                 target_length: Optional[int] = None,
                                 method: str = "truncate") -> Dict[str, pd.DataFrame]:
    """
    Normalize time series to uniform length using specified method.
    
    This utility function handles variable-length time series by either truncating
    longer series or padding shorter ones to achieve uniform length across all
    time series in the dataset.
    
    Args:
        ts_dict (Dict[str, pd.DataFrame]): Dictionary of time series DataFrames
        target_length (Optional[int]): Desired length. If None, uses min/max of existing lengths
        method (str): Normalization method - "truncate" or "pad"
                     - "truncate": Cut longer series to target_length (uses minimum length if target_length is None)
                     - "pad": Extend shorter series by repeating last value (uses maximum length if target_length is None)
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary with normalized time series of uniform length
    
    Example:
        >>> ts_data = {"series1": pd.DataFrame([1,2,3]), "series2": pd.DataFrame([1,2])}
        >>> normalized = normalize_time_series_length(ts_data, method="truncate")
        >>> # All series will have length 2 (minimum)
    
    Note:
        - Padding uses forward-fill strategy (repeats last observed value)
        - Original DataFrames are not modified (copies are created)
        - Empty input dictionary returns empty dictionary
    """
    if not ts_dict:
        return ts_dict
    
    lengths = [len(df) for df in ts_dict.values()]
    
    if target_length is None:
        if method == "truncate":
            target_length = min(lengths)
        else:
            target_length = max(lengths)
    
    normalized_dict = {}
    
    for series_id, df in ts_dict.items():
        if method == "truncate":
            normalized_dict[series_id] = df.iloc[:target_length].copy()
        elif method == "pad":
            if len(df) < target_length:
                last_row = df.iloc[-1:].copy()
                padding_rows = pd.concat([last_row] * (target_length - len(df)), ignore_index=True)
                normalized_dict[series_id] = pd.concat([df, padding_rows], ignore_index=True)
            else:
                normalized_dict[series_id] = df.copy()
        else:
            normalized_dict[series_id] = df.copy()
    
    return normalized_dict

if __name__ == "__main__":
    loader = TimeSeriesDataLoader("TimeSeries")
    
    print("Data Summary:")
    summary = loader.get_data_summary()
    
    print("\nConditional variants:", summary["conditional_variants"])
    print("Unconditional variants:", summary["unconditional_variants"])
    
    print("\nValidation Results:")
    for data_type, variants in summary["validation"].items():
        print(f"\n{data_type.upper()}:")
        for variant, stats in variants.items():
            print(f"  {variant}: {stats}")