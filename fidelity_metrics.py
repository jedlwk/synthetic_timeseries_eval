"""
TSGBench Core Fidelity Metrics Module

This module implements the exact TSGBench fidelity evaluation metrics as specified
in the TSGBench: Time Series Generation Benchmark (VLDB 2024) paper.

Core TSGBench Feature-based Measures:
- Marginal Distribution Difference (MDD): Histogram-based distributional divergence
- Autocorrelation Difference (ACD): Temporal dependency structure preservation  
- Skewness Difference (SD): Third statistical moment differences
- Kurtosis Difference (KD): Fourth statistical moment differences

Core TSGBench Distance-based Measures:
- Dynamic Time Warping (DTW): Optimal sequence alignment distance
- Euclidean Distance (ED): Point-wise geometric similarity

Implementation follows exact TSGBench methodology from the original paper:
"TSGBench: Time Series Generation Benchmark" (VLDB 2024)

References:
- Ang, Yihao, et al. "TSGBench: Time Series Generation Benchmark." VLDB Endowment, 2024.
- GitHub: https://github.com/YihaoAng/TSGBench

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp, wasserstein_distance, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')

class FidelityMetrics:
    """
    TSGBench fidelity evaluation metrics implementation.
    
    Implements the exact feature-based and distance-based measures from
    TSGBench: Time Series Generation Benchmark (VLDB 2024).
    
    TSGBench Feature-based Measures:
    - MDD: Marginal Distribution Difference using histogram densities
    - ACD: Autocorrelation Difference for temporal structure
    - SD: Skewness Difference for distributional asymmetry
    - KD: Kurtosis Difference for distribution tail analysis
    
    TSGBench Distance-based Measures:
    - DTW: Dynamic Time Warping for sequence similarity
    - ED: Euclidean Distance for point-wise comparison
    
    Example:
        >>> evaluator = FidelityMetrics()
        >>> results = evaluator.compute_all_fidelity_metrics(original, synthetic)
        >>> print(f"MDD: {results['mdd_mean']:.3f}")
    """
    def __init__(self):
        """Initialize TSGBench fidelity evaluator with scaler for handling scale differences."""
        self.scaler = StandardScaler()
    
    def marginal_distribution_difference(self, original_data: Dict[str, pd.DataFrame],
                                       synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute TSGBench Marginal Distribution Difference (MDD) using histogram-based approach.
        
        TSGBench MDD Methodology:
        - Calculate histogram densities for original and synthetic data
        - Compute absolute differences between histogram bins
        - Average differences across all features
        
        Returns histogram-based distributional divergence as in TSGBench paper.
        """
        original_matrix = self._dict_to_matrix(original_data)
        if original_matrix.shape[0] == 0:
            return {"mdd_mean": 0.0, "mdd_std": 0.0, "mdd_max": 0.0}
            
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if synthetic_matrix.shape[0] == 0:
            return {"mdd_mean": 1.0, "mdd_std": 0.0, "mdd_max": 1.0}
        
        try:
            # Fit scaler on original data and transform both datasets
            original_scaled = self.scaler.fit_transform(original_matrix)
            synthetic_scaled = self.scaler.transform(synthetic_matrix)
        except Exception:
            # Fallback to raw data if scaling fails
            original_scaled = original_matrix
            synthetic_scaled = synthetic_matrix
        
        mdd_scores = []
        
        for col_idx in range(min(original_scaled.shape[1], synthetic_scaled.shape[1])):
            orig_col = original_scaled[:, col_idx]
            synth_col = synthetic_scaled[:, col_idx]
            
            orig_col = orig_col[~np.isnan(orig_col)]
            synth_col = synth_col[~np.isnan(synth_col)]
            
            if len(orig_col) > 0 and len(synth_col) > 0:
                mdd_score = self._calculate_histogram_mdd(orig_col, synth_col)
                mdd_scores.append(mdd_score)
        
        if not mdd_scores:
            return {"mdd_mean": 1.0, "mdd_std": 0.0, "mdd_max": 1.0}
        
        return {
            "mdd_mean": np.mean(mdd_scores),
            "mdd_std": np.std(mdd_scores),
            "mdd_max": np.max(mdd_scores)
        }
    
    def autocorrelation_difference(self, original_data: Dict[str, pd.DataFrame],
                                 synthetic_data: Dict[str, pd.DataFrame],
                                 max_lag: int = 10) -> Dict[str, float]:
        original_autocorrs = []
        synthetic_autocorrs = []
        
        for series_id, df in original_data.items():
            if df.empty:
                continue
            autocorrs = self._calculate_autocorrelation(df, max_lag)
            original_autocorrs.append(autocorrs)
        
        for series_id, df in synthetic_data.items():
            if df.empty:
                continue
            autocorrs = self._calculate_autocorrelation(df, max_lag)
            synthetic_autocorrs.append(autocorrs)
        
        if not original_autocorrs or not synthetic_autocorrs:
            return {"acd_mean": 1.0, "acd_std": 0.0, "acd_max": 1.0}
        
        min_samples = min(len(original_autocorrs), len(synthetic_autocorrs))
        original_autocorrs = original_autocorrs[:min_samples]
        synthetic_autocorrs = synthetic_autocorrs[:min_samples]
        
        differences = []
        for orig, synth in zip(original_autocorrs, synthetic_autocorrs):
            diff = np.mean(np.abs(np.array(orig) - np.array(synth)))
            differences.append(diff)
        
        return {
            "acd_mean": np.mean(differences),
            "acd_std": np.std(differences),
            "acd_max": np.max(differences)
        }
    
    def statistical_moments_difference(self, original_data: Dict[str, pd.DataFrame],
                                     synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        original_matrix = self._dict_to_matrix(original_data)
        if original_matrix.shape[0] == 0:
            return {"skewness_diff": 0.0, "kurtosis_diff": 0.0, "moment_score": 0.0}
            
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if synthetic_matrix.shape[0] == 0:
            return {"skewness_diff": 1.0, "kurtosis_diff": 1.0, "moment_score": 0.0}
        
        skewness_diffs = []
        kurtosis_diffs = []
        
        for col_idx in range(min(original_matrix.shape[1], synthetic_matrix.shape[1])):
            orig_col = original_matrix[:, col_idx]
            synth_col = synthetic_matrix[:, col_idx]
            
            orig_col = orig_col[~np.isnan(orig_col)]
            synth_col = synth_col[~np.isnan(synth_col)]
            
            if len(orig_col) > 3 and len(synth_col) > 3:
                # Calculate skewness with NaN handling
                orig_skew = stats.skew(orig_col)
                synth_skew = stats.skew(synth_col)
                if not (np.isnan(orig_skew) or np.isnan(synth_skew)):
                    skewness_diffs.append(abs(orig_skew - synth_skew))
                
                # Calculate kurtosis with NaN handling
                orig_kurt = stats.kurtosis(orig_col)
                synth_kurt = stats.kurtosis(synth_col)
                if not (np.isnan(orig_kurt) or np.isnan(synth_kurt)):
                    kurtosis_diffs.append(abs(orig_kurt - synth_kurt))
        
        skewness_diff = np.mean(skewness_diffs) if skewness_diffs else 1.0
        kurtosis_diff = np.mean(kurtosis_diffs) if kurtosis_diffs else 1.0
        
        moment_score = 1.0 - (skewness_diff + kurtosis_diff) / 2.0
        moment_score = max(0.0, moment_score)
        
        return {
            "skewness_diff": skewness_diff,
            "kurtosis_diff": kurtosis_diff,
            "moment_score": moment_score
        }
    
    def dynamic_time_warping_distance(self, original_data: Dict[str, pd.DataFrame],
                                    synthetic_data: Dict[str, pd.DataFrame],
                                    sample_size: int = 20) -> Dict[str, float]:
        if not original_data or not synthetic_data:
            return {"dtw_mean": 1.0, "dtw_std": 0.0, "dtw_min": 1.0}
        
        orig_series = list(original_data.values())[:sample_size]
        synth_series = list(synthetic_data.values())[:sample_size]
        
        dtw_distances = []
        
        min_pairs = min(len(orig_series), len(synth_series))
        
        for i in range(min_pairs):
            orig_df = orig_series[i]
            synth_df = synth_series[i]
            
            if orig_df.empty or synth_df.empty:
                continue
                
            orig_numeric = orig_df.select_dtypes(include=[np.number])
            synth_numeric = synth_df.select_dtypes(include=[np.number])
            
            if orig_numeric.empty or synth_numeric.empty:
                continue
            
            for col in orig_numeric.columns:
                if col in synth_numeric.columns:
                    orig_series_data = orig_numeric[col].dropna().values
                    synth_series_data = synth_numeric[col].dropna().values
                    
                    if len(orig_series_data) > 1 and len(synth_series_data) > 1:
                        try:
                            distance = dtw.distance(orig_series_data, synth_series_data)
                            if not np.isnan(distance) and not np.isinf(distance):
                                normalized_distance = distance / max(len(orig_series_data), len(synth_series_data))
                                dtw_distances.append(normalized_distance)
                        except:
                            continue
        
        if not dtw_distances:
            return {"dtw_mean": 1.0, "dtw_std": 0.0, "dtw_min": 1.0}
        
        return {
            "dtw_mean": np.mean(dtw_distances),
            "dtw_std": np.std(dtw_distances),
            "dtw_min": np.min(dtw_distances)
        }
    
    def euclidean_distance_analysis(self, original_data: Dict[str, pd.DataFrame],
                                  synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        original_matrix = self._dict_to_matrix(original_data)
        if original_matrix.shape[0] == 0:
            return {"euclidean_mean": 1.0, "euclidean_std": 0.0, "euclidean_min": 1.0}
            
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if synthetic_matrix.shape[0] == 0:
            return {"euclidean_mean": 1.0, "euclidean_std": 0.0, "euclidean_min": 1.0}
        
        distances = []
        min_samples = min(original_matrix.shape[0], synthetic_matrix.shape[0])
        
        for i in range(min_samples):
            orig_sample = original_matrix[i]
            synth_sample = synthetic_matrix[i]
            
            orig_sample = orig_sample[~np.isnan(orig_sample)]
            synth_sample = synth_sample[~np.isnan(synth_sample)]
            
            if len(orig_sample) > 0 and len(synth_sample) > 0:
                min_len = min(len(orig_sample), len(synth_sample))
                distance = euclidean(orig_sample[:min_len], synth_sample[:min_len])
                normalized_distance = distance / np.sqrt(min_len)
                distances.append(normalized_distance)
        
        if not distances:
            return {"euclidean_mean": 1.0, "euclidean_std": 0.0, "euclidean_min": 1.0}
        
        return {
            "euclidean_mean": np.mean(distances),
            "euclidean_std": np.std(distances),
            "euclidean_min": np.min(distances)
        }
    
    def wasserstein_distance_analysis(self, original_data: Dict[str, pd.DataFrame],
                                    synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute Wasserstein (Earth Mover's) distances between feature distributions.
        
        This metric evaluates distributional similarity using optimal transport theory,
        measuring the minimum cost to transform one distribution into another. It provides
        a theoretically grounded assessment of how well synthetic data preserves the
        underlying probability distributions.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - wasserstein_mean: Average Wasserstein distance [0, ∞)
                - wasserstein_std: Standard deviation of distances
                - wasserstein_min: Minimum distance (best distribution match)
        
        Note:
            - Based on optimal transport theory (Villani 2009)
            - Also known as Earth Mover's Distance (EMD)
            - More sensitive to tail differences than KS-test
            - Computed feature-wise across all dimensions
            - Lower values indicate better distributional fidelity
            - Complements marginal distribution difference metrics
        
        Example:
            >>> wass = evaluator.wasserstein_distance_analysis(original, synthetic)
            >>> if wass['wasserstein_mean'] > 0.2:
            ...     print("Significant distributional differences detected")
        """
        original_matrix = self._dict_to_matrix(original_data)
        if original_matrix.shape[0] == 0:
            return {"wasserstein_mean": 1.0, "wasserstein_std": 0.0, "wasserstein_min": 1.0}
            
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if synthetic_matrix.shape[0] == 0:
            return {"wasserstein_mean": 1.0, "wasserstein_std": 0.0, "wasserstein_min": 1.0}
        
        try:
            # Fit scaler on original data and transform both datasets
            original_scaled = self.scaler.fit_transform(original_matrix)
            synthetic_scaled = self.scaler.transform(synthetic_matrix)
        except Exception:
            # Fallback to raw data if scaling fails
            original_scaled = original_matrix
            synthetic_scaled = synthetic_matrix
        
        distances = []
        
        for col_idx in range(min(original_scaled.shape[1], synthetic_scaled.shape[1])):
            orig_col = original_scaled[:, col_idx]
            synth_col = synthetic_scaled[:, col_idx]
            
            orig_col = orig_col[~np.isnan(orig_col)]
            synth_col = synth_col[~np.isnan(synth_col)]
            
            if len(orig_col) > 0 and len(synth_col) > 0:
                try:
                    distance = wasserstein_distance(orig_col, synth_col)
                    if not np.isnan(distance) and not np.isinf(distance):
                        distances.append(distance)
                except:
                    continue
        
        if not distances:
            return {"wasserstein_mean": 1.0, "wasserstein_std": 0.0, "wasserstein_min": 1.0}
        
        return {
            "wasserstein_mean": np.mean(distances),
            "wasserstein_std": np.std(distances),
            "wasserstein_min": np.min(distances)
        }
    
    def compute_all_fidelity_metrics(self, original_data: Dict[str, pd.DataFrame],
                                   synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute comprehensive fidelity evaluation across all implemented metrics.
        
        This is the main evaluation method that computes all fidelity metrics and
        provides an overall fidelity score weighted according to TSGBench methodology.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
        
        Returns:
            Dict[str, float]: Complete fidelity evaluation containing:
                - mdd_mean/std/max: Marginal distribution differences
                - acd_mean/std/max: Autocorrelation differences
                - skewness_diff/kurtosis_diff/moment_score: Statistical moments
                - dtw_mean/std/min: Dynamic time warping distances
                - euclidean_mean/std/min: Euclidean distances
                - wasserstein_mean/std/min: Wasserstein distances
                - overall_fidelity_score: Weighted average across key metrics
        
        Note:
            - Overall score emphasizes distributional similarity (MDD, Wasserstein)
            - Temporal structure preservation (DTW, autocorrelation) weighted highly
            - Statistical moments provide shape characteristic evaluation
        
        Example:
            >>> results = evaluator.compute_all_fidelity_metrics(original, synthetic)
            >>> print(f"Overall fidelity: {results['overall_fidelity_score']:.2%}")
            >>> if results['mdd_mean'] > 0.3:
            ...     print("Alert: Poor distributional fidelity")
        """
        results = {}
        
        mdd_results = self.marginal_distribution_difference(original_data, synthetic_data)
        results.update(mdd_results)
        
        acd_results = self.autocorrelation_difference(original_data, synthetic_data)
        results.update(acd_results)
        
        moment_results = self.statistical_moments_difference(original_data, synthetic_data)
        results.update(moment_results)
        
        dtw_results = self.dynamic_time_warping_distance(original_data, synthetic_data)
        results.update(dtw_results)
        
        euclidean_results = self.euclidean_distance_analysis(original_data, synthetic_data)
        results.update(euclidean_results)
        
        wasserstein_results = self.wasserstein_distance_analysis(original_data, synthetic_data)
        results.update(wasserstein_results)
        
        fidelity_score = 1.0 - np.mean([
            results["mdd_mean"],
            results["acd_mean"],
            1.0 - results["moment_score"],
            min(results["dtw_mean"], 1.0),
            min(results["euclidean_mean"], 1.0),
            min(results["wasserstein_mean"], 1.0)
        ])
        
        results["overall_fidelity_score"] = max(0.0, fidelity_score)
        
        return results
    
    def compute_column_fidelity_metrics(self, original_data: Dict[str, pd.DataFrame],
                                      synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """
        Compute fidelity metrics for each individual column across all time series.
        
        Returns:
            Dict[str, Dict[str, float]]: Column name -> fidelity metrics for that column
        """
        column_results = {}
        
        # Get all unique column names from both datasets
        all_columns = set()
        for df in list(original_data.values()) + list(synthetic_data.values()):
            if not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                all_columns.update(numeric_cols)
        
        # Filter out non-financial columns if needed
        financial_columns = []
        for col in all_columns:
            if col.lower() not in ['openint']:  # Skip OpenInt as requested
                financial_columns.append(col)
        
        for column in financial_columns:
            column_results[column] = self._compute_single_column_fidelity(
                original_data, synthetic_data, column
            )
        
        return column_results
    
    def _compute_single_column_fidelity(self, original_data: Dict[str, pd.DataFrame],
                                      synthetic_data: Dict[str, pd.DataFrame], 
                                      column: str) -> Dict[str, float]:
        """Compute fidelity metrics for a single column across all time series."""
        orig_values = []
        synth_values = []
        
        # Collect all values for this column from all time series
        # Handle both exact key matches and mismatched keys (use all available data)
        orig_keys = list(original_data.keys())
        synth_keys = list(synthetic_data.keys())
        
        # First try exact key matching
        exact_matches = set(orig_keys) & set(synth_keys)
        if exact_matches:
            for key in exact_matches:
                orig_df = original_data[key]
                synth_df = synthetic_data[key]
                
                if column in orig_df.columns and column in synth_df.columns:
                    orig_col = orig_df[column].dropna().values
                    synth_col = synth_df[column].dropna().values
                    
                    if len(orig_col) > 0 and len(synth_col) > 0:
                        orig_values.extend(orig_col)
                        synth_values.extend(synth_col)
        else:
            # If no exact matches, collect all data from both datasets
            # This handles cases where file names don't match (e.g., stock symbols vs indices)
            for key in orig_keys:
                orig_df = original_data[key]
                if column in orig_df.columns:
                    orig_col = orig_df[column].dropna().values
                    if len(orig_col) > 0:
                        orig_values.extend(orig_col)
            
            for key in synth_keys:
                synth_df = synthetic_data[key]
                if column in synth_df.columns:
                    synth_col = synth_df[column].dropna().values
                    if len(synth_col) > 0:
                        synth_values.extend(synth_col)
        
        if len(orig_values) == 0 or len(synth_values) == 0:
            return {"column_fidelity_score": 0.0, "mdd": 1.0, "moment_similarity": 0.0}
        
        orig_values = np.array(orig_values)
        synth_values = np.array(synth_values)
        
        # Marginal distribution difference (KS test)
        try:
            from scipy import stats
            ks_stat, _ = stats.ks_2samp(orig_values, synth_values)
            mdd_score = max(0.0, 1.0 - ks_stat)
        except:
            mdd_score = 0.0
        
        # Statistical moments similarity
        orig_moments = [np.mean(orig_values), np.std(orig_values), 
                       stats.skew(orig_values), stats.kurtosis(orig_values)]
        synth_moments = [np.mean(synth_values), np.std(synth_values),
                        stats.skew(synth_values), stats.kurtosis(synth_values)]
        
        moment_diffs = [abs(o - s) / (abs(o) + 1e-10) for o, s in zip(orig_moments, synth_moments)]
        moment_similarity = max(0.0, 1.0 - np.mean(moment_diffs))
        
        # Overall column fidelity score
        column_fidelity_score = 0.6 * mdd_score + 0.4 * moment_similarity
        
        return {
            "column_fidelity_score": column_fidelity_score,
            "mdd": mdd_score,
            "moment_similarity": moment_similarity
        }
    
    def _dict_to_matrix(self, data_dict: Dict[str, pd.DataFrame], target_length: int = None) -> np.ndarray:
        if not data_dict:
            return np.array([])
        
        matrices = []
        lengths = []
        
        for series_id, df in data_dict.items():
            if df.empty:
                continue
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                flattened = numeric_df.values.flatten()
                matrices.append(flattened)
                lengths.append(len(flattened))
        
        if not matrices:
            return np.array([])
        
        if target_length is None:
            target_length = min(lengths)  # Use minimum length for overlapping data only
        
        # Use only overlapping timesteps - no padding, only truncation
        normalized_matrices = []
        for matrix in matrices:
            if len(matrix) >= target_length:
                normalized_matrices.append(matrix[:target_length])
            # Skip matrices that are shorter than target_length - no padding applied
        
        return np.array(normalized_matrices)
    
    def _calculate_autocorrelation(self, df: pd.DataFrame, max_lag: int = 10) -> List[float]:
        autocorrs = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > max_lag:
                col_autocorrs = []
                for lag in range(1, min(max_lag + 1, len(series))):
                    try:
                        autocorr = series.autocorr(lag=lag)
                        if pd.isna(autocorr):
                            autocorr = 0.0
                        col_autocorrs.append(autocorr)
                    except:
                        col_autocorrs.append(0.0)
                
                while len(col_autocorrs) < max_lag:
                    col_autocorrs.append(0.0)
                
                autocorrs.extend(col_autocorrs)
            else:
                autocorrs.extend([0.0] * max_lag)
        
        return autocorrs if autocorrs else [0.0]
    
    def _calculate_histogram_mdd(self, orig_data: np.ndarray, synth_data: np.ndarray, bins: int = 50) -> float:
        """
        Calculate TSGBench histogram-based Marginal Distribution Difference.
        
        TSGBench methodology: compute histogram densities and measure absolute differences.
        """
        try:
            # Determine common range for both datasets
            min_val = min(np.min(orig_data), np.min(synth_data))
            max_val = max(np.max(orig_data), np.max(synth_data))
            
            # Handle edge case where all values are identical
            if min_val == max_val:
                return 0.0
            
            # Create histogram bins
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            # Calculate normalized histograms (densities)
            orig_hist, _ = np.histogram(orig_data, bins=bin_edges, density=True)
            synth_hist, _ = np.histogram(synth_data, bins=bin_edges, density=True)
            
            # TSGBench MDD: absolute difference between histogram densities
            bin_width = (max_val - min_val) / bins
            mdd = np.sum(np.abs(orig_hist - synth_hist)) * bin_width
            
            return mdd
            
        except Exception:
            # Fallback to KS-test if histogram calculation fails
            try:
                ks_stat, _ = ks_2samp(orig_data, synth_data)
                return ks_stat
            except:
                return 1.0

if __name__ == "__main__":
    from data_loader import TimeSeriesDataLoader
    
    loader = TimeSeriesDataLoader("TimeSeries")
    fidelity_evaluator = FidelityMetrics()
    
    print("Testing Fidelity Metrics...")
    
    _, original_cond = loader.load_conditional_data("original")
    _, synthetic_cond = loader.load_conditional_data("tsv2")
    
    if original_cond and synthetic_cond:
        fidelity_results = fidelity_evaluator.compute_all_fidelity_metrics(
            original_cond, synthetic_cond
        )
        
        print("\nConditional Generation Fidelity Results:")
        for metric, value in fidelity_results.items():
            print(f"  {metric}: {value:.4f}")
    
    original_uncond = loader.load_unconditional_data("original")
    synthetic_uncond = loader.load_unconditional_data("tsv2")
    
    if original_uncond and synthetic_uncond:
        fidelity_results = fidelity_evaluator.compute_all_fidelity_metrics(
            original_uncond, synthetic_uncond
        )
        
        print("\nUnconditional Generation Fidelity Results:")
        for metric, value in fidelity_results.items():
            print(f"  {metric}: {value:.4f}")