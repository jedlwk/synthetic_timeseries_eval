"""
TSGBench-Aligned Diversity Metrics Module

This module implements diversity evaluation metrics following TSGBench methodology
for standardized evaluation of synthetic time series generation quality.

Core TSGBench diversity evaluation approach:
- Statistical Diversity: Variance, range, and entropy preservation across features
- Coverage Analysis: Proportion of original data space covered by synthetic samples  
- Anti-Mode Collapse: Uniqueness score measuring generation variety
- Temporal Patterns: Autocorrelation diversity for time series characteristics

Mixed Domain Support:
- Conditional Generation: Generic time series (col1, col2, col3) with equal weighting
- Unconditional Generation: Financial OHLC-V data with domain-specific weighting (OHLC: 70%, Volume: 30%)

References:
- Ang, Yihao, et al. "TSGBench: Time Series Generation Benchmark." VLDB 2024 (Primary methodology)

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import warnings

class DiversityMetrics:
    """
    TSGBench-aligned diversity evaluation for synthetic time series data.
    
    Implements core TSGBench diversity metrics with domain-specific extensions
    for both generic time series (conditional) and financial data (unconditional).
    
    TSGBench Core Metrics:
    - Statistical diversity (variance, range, entropy ratios)
    - Coverage analysis with distance thresholds
    - Anti-mode collapse detection (uniqueness scoring)
    - Temporal pattern preservation (autocorrelation diversity)
    
    Domain Adaptations:
    - Generic Data: Equal weighting across col1, col2, col3
    - Financial Data: OHLC price features (70%) + Volume (30%) weighting
    
    Example:
        >>> evaluator = DiversityMetrics()
        >>> results = evaluator.compute_all_diversity_metrics(original, synthetic)
        >>> print(f"Overall diversity: {results['overall_diversity_score']:.3f}")
    """
    def __init__(self):
        """
        Initialize the DiversityMetrics evaluator.
        
        Sets up the internal StandardScaler for consistent data normalization
        across all diversity metrics.
        """
        self.scaler = StandardScaler()
    
    def coverage_ratio(self, original_data: Dict[str, pd.DataFrame], 
                      synthetic_data: Dict[str, pd.DataFrame],
                      coverage_threshold: float = 0.5) -> float:
        """
        Calculate the proportion of original data space covered by synthetic data.
        
        This metric measures how well synthetic data covers the original data distribution
        by computing the fraction of original points that have at least one synthetic
        point within a specified distance threshold. Higher values indicate better
        coverage and reduced mode collapse.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
            coverage_threshold (float): Maximum distance for considering a point "covered"
                                       Default 0.5 based on normalized data scale and empirical validation
        
        Returns:
            float: Coverage ratio in [0, 1], where 1.0 means perfect coverage
        
        Note:
            - Uses StandardScaler normalization before distance computation
            - Employs Euclidean distance in flattened feature space
            - Based on TSGBench coverage ratio implementation
        
        Example:
            >>> coverage = evaluator.coverage_ratio(original_data, synthetic_data)
            >>> print(f"Coverage: {coverage:.1%}")  # e.g., "Coverage: 85.3%"
        """
        original_matrix = self._dict_to_matrix(original_data)
        
        if original_matrix.shape[0] == 0:
            return 0.0
        
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if synthetic_matrix.shape[0] == 0:
            return 0.0
        
        original_scaled = self.scaler.fit_transform(original_matrix)
        synthetic_scaled = self.scaler.transform(synthetic_matrix)
        
        coverage_count = 0
        total_original = original_scaled.shape[0]
        
        for orig_point in original_scaled:
            distances = np.linalg.norm(synthetic_scaled - orig_point, axis=1)
            min_distance = np.min(distances)
            
            if min_distance <= coverage_threshold:
                coverage_count += 1
        
        return coverage_count / total_original
    
    def uniqueness_score(self, synthetic_data: Dict[str, pd.DataFrame],
                        uniqueness_threshold: float = 1e-6) -> float:
        """
        Calculate the uniqueness score of synthetic data samples.
        
        This metric measures the proportion of unique synthetic samples by computing
        pairwise distances and identifying samples that are sufficiently different
        from each other. Higher scores indicate better sample diversity and reduced
        mode collapse in the generative model.
        
        Args:
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
            uniqueness_threshold (float): Minimum distance for considering samples unique
                                         Default 1e-6 for numerical precision
        
        Returns:
            float: Uniqueness score in [0, 1], where 1.0 means all samples are unique
        
        Note:
            - Uses pairwise Euclidean distances in flattened feature space
            - Threshold should be adjusted based on data scale and precision requirements
            - Returns 1.0 for single samples (trivially unique)
        
        Example:
            >>> uniqueness = evaluator.uniqueness_score(synthetic_data)
            >>> if uniqueness < 0.9:
            ...     print("Warning: Potential mode collapse detected")
        """
        synthetic_matrix = self._dict_to_matrix(synthetic_data)
        
        if synthetic_matrix.shape[0] <= 1:
            return 1.0
        
        distances = pdist(synthetic_matrix)
        unique_pairs = np.sum(distances > uniqueness_threshold)
        total_pairs = len(distances)
        
        return unique_pairs / total_pairs if total_pairs > 0 else 1.0
    
    def statistical_diversity(self, original_data: Dict[str, pd.DataFrame],
                            synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Evaluate statistical diversity through variance, range, and entropy preservation.
        
        This method computes how well synthetic data preserves the statistical properties
        of the original data across multiple dimensions: variance (spread), range (extent),
        and entropy (information content). These metrics together provide a comprehensive
        view of distributional diversity preservation.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - variance_ratio: Ratio of synthetic to original variance (capped at 1.0)
                - range_ratio: Ratio of synthetic to original range (capped at 1.0)
                - entropy_ratio: Ratio of synthetic to original entropy (capped at 1.0)
        
        Note:
            - Uses symmetric ratios: min(synth/orig, orig/synth) to penalize both over/under representation
            - Handles zero-variance features gracefully
            - Entropy computed using histogram-based discretization
        
        Example:
            >>> stats = evaluator.statistical_diversity(original_data, synthetic_data)
            >>> print(f"Variance preserved: {stats['variance_ratio']:.1%}")
        """
        original_matrix = self._dict_to_matrix(original_data)
        
        if original_matrix.shape[0] == 0:
            return {"variance_ratio": 0.0, "range_ratio": 0.0, "entropy_ratio": 0.0}
        
        target_length = original_matrix.shape[1]
        synthetic_matrix = self._dict_to_matrix(synthetic_data, target_length)
        
        if original_matrix.shape[0] == 0 or synthetic_matrix.shape[0] == 0:
            return {"variance_ratio": 0.0, "range_ratio": 0.0, "entropy_ratio": 0.0}
        
        results = {}
        
        orig_var = np.var(original_matrix, axis=0)
        synth_var = np.var(synthetic_matrix, axis=0)
        
        variance_ratios = []
        for i in range(len(orig_var)):
            if orig_var[i] > 0:
                variance_ratios.append(min(synth_var[i] / orig_var[i], orig_var[i] / synth_var[i]))
            else:
                variance_ratios.append(1.0 if synth_var[i] == 0 else 0.0)
        
        results["variance_ratio"] = np.mean(variance_ratios)
        
        orig_range = np.ptp(original_matrix, axis=0)
        synth_range = np.ptp(synthetic_matrix, axis=0)
        
        range_ratios = []
        for i in range(len(orig_range)):
            if orig_range[i] > 0:
                range_ratios.append(min(synth_range[i] / orig_range[i], orig_range[i] / synth_range[i]))
            else:
                range_ratios.append(1.0 if synth_range[i] == 0 else 0.0)
        
        results["range_ratio"] = np.mean(range_ratios)
        
        entropy_ratios = []
        for col_idx in range(original_matrix.shape[1]):
            orig_col = original_matrix[:, col_idx]
            synth_col = synthetic_matrix[:, col_idx]
            
            orig_entropy = self._calculate_entropy(orig_col)
            synth_entropy = self._calculate_entropy(synth_col)
            
            if orig_entropy > 0:
                entropy_ratios.append(min(synth_entropy / orig_entropy, orig_entropy / synth_entropy))
            else:
                entropy_ratios.append(1.0 if synth_entropy == 0 else 0.0)
        
        results["entropy_ratio"] = np.mean(entropy_ratios)
        
        return results
    
    def temporal_pattern_diversity(self, original_data: Dict[str, pd.DataFrame],
                                 synthetic_data: Dict[str, pd.DataFrame],
                                 max_lag: int = 10) -> Dict[str, float]:
        """
        Assess diversity in temporal patterns through autocorrelation and trend analysis.
        
        This method evaluates how well synthetic data preserves the variety of temporal
        patterns present in the original data. It analyzes both autocorrelation structures
        (memory effects) and trend characteristics (directional changes) across the dataset.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
            max_lag (int): Maximum lag for autocorrelation computation (default: 10)
        
        Returns:
            Dict[str, float]: Dictionary containing:
                - autocorr_diversity: Diversity in autocorrelation patterns [0, 1]
                - trend_diversity: Diversity in trend characteristics [0, 1]
        
        Note:
            - Autocorrelation patterns capture temporal dependencies and memory effects
            - Trend features include linear slope and second-order derivatives
            - Higher values indicate better preservation of temporal pattern variety
        
        Example:
            >>> temporal = evaluator.temporal_pattern_diversity(original, synthetic)
            >>> if temporal['autocorr_diversity'] < 0.5:
            ...     print("Warning: Poor temporal memory preservation")
        """
        results = {}
        
        if not original_data or not synthetic_data:
            return {"autocorr_diversity": 0.0, "trend_diversity": 0.0}
        
        orig_autocorr_patterns = []
        synth_autocorr_patterns = []
        
        orig_trend_patterns = []
        synth_trend_patterns = []
        
        for series_id, df in original_data.items():
            if df.empty:
                continue
            autocorrs = self._calculate_autocorrelation(df, max_lag)
            trends = self._calculate_trend_features(df)
            orig_autocorr_patterns.append(autocorrs)
            orig_trend_patterns.append(trends)
        
        for series_id, df in synthetic_data.items():
            if df.empty:
                continue
            autocorrs = self._calculate_autocorrelation(df, max_lag)
            trends = self._calculate_trend_features(df)
            synth_autocorr_patterns.append(autocorrs)
            synth_trend_patterns.append(trends)
        
        if orig_autocorr_patterns and synth_autocorr_patterns:
            orig_autocorr_var = np.var(orig_autocorr_patterns, axis=0)
            synth_autocorr_var = np.var(synth_autocorr_patterns, axis=0)
            
            autocorr_ratios = []
            for i in range(len(orig_autocorr_var)):
                if orig_autocorr_var[i] > 0:
                    ratio = min(synth_autocorr_var[i] / orig_autocorr_var[i], 
                              orig_autocorr_var[i] / synth_autocorr_var[i])
                    autocorr_ratios.append(ratio)
                else:
                    autocorr_ratios.append(1.0 if synth_autocorr_var[i] == 0 else 0.0)
            
            results["autocorr_diversity"] = np.mean(autocorr_ratios)
        else:
            results["autocorr_diversity"] = 0.0
        
        if orig_trend_patterns and synth_trend_patterns:
            orig_trend_var = np.var(orig_trend_patterns, axis=0)
            synth_trend_var = np.var(synth_trend_patterns, axis=0)
            
            trend_ratios = []
            for i in range(len(orig_trend_var)):
                if orig_trend_var[i] > 0:
                    ratio = min(synth_trend_var[i] / orig_trend_var[i], 
                              orig_trend_var[i] / synth_trend_var[i])
                    trend_ratios.append(ratio)
                else:
                    trend_ratios.append(1.0 if synth_trend_var[i] == 0 else 0.0)
            
            results["trend_diversity"] = np.mean(trend_ratios)
        else:
            results["trend_diversity"] = 0.0
        
        return results
    
    def compute_all_diversity_metrics(self, original_data: Dict[str, pd.DataFrame],
                                    synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Compute comprehensive diversity evaluation with both overall and individual column scores.
        
        This method provides two levels of analysis:
        1. Individual column scores for each financial feature (Open, High, Low, Close, Volume)
        2. Overall aggregated scores using financial domain weighting
        
        Global normalization ensures fair comparison across all columns and datasets.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): Original time series data
            synthetic_data (Dict[str, pd.DataFrame]): Synthetic time series data
        
        Returns:
            Dict[str, float]: Complete diversity evaluation containing:
                - Individual column scores: {column}_diversity_score, {column}_variance_ratio, etc.
                - Overall aggregated scores: diversity_score, variance_ratio, etc.
                - Global metrics: coverage_ratio, uniqueness_score
                - Legacy compatibility metrics
        
        Example:
            >>> results = evaluator.compute_all_diversity_metrics(original, synthetic)
            >>> print(f"Overall diversity: {results['overall_diversity_score']:.2%}")
            >>> print(f"Open diversity: {results['Open_diversity_score']:.2%}")
            >>> print(f"Volume diversity: {results['Volume_diversity_score']:.2%}")
        """
        results = {}
        
        # Get globally normalized column-wise matrices
        orig_columns, synth_columns, normalization_stats = self._dict_to_normalized_column_matrices(
            original_data, synthetic_data
        )
        
        if not orig_columns or not synth_columns:
            # Fallback to legacy method if column processing fails
            return self._compute_legacy_diversity(original_data, synthetic_data)
        
        # Store normalization info for transparency
        results["_normalization_stats"] = normalization_stats
        
        # Individual column diversity metrics
        column_scores = {}
        for column in orig_columns.keys():
            if column in synth_columns:
                col_scores = self._compute_column_diversity(
                    orig_columns[column], synth_columns[column], column
                )
                column_scores[column] = col_scores
                
                # Add individual column results with clear naming
                for metric, value in col_scores.items():
                    results[f"{column}_{metric}"] = value
                
                # Add overall column diversity score
                results[f"{column}_diversity_score"] = np.mean([
                    col_scores.get("variance_ratio", 0.0),
                    col_scores.get("range_ratio", 0.0), 
                    col_scores.get("entropy_ratio", 0.0),
                    col_scores.get("autocorr_diversity", 0.0),
                    col_scores.get("coverage", 0.0)
                ])
        
        # Global metrics (cross-column)
        results["coverage_ratio"] = self.coverage_ratio(original_data, synthetic_data)
        results["uniqueness_score"] = self.uniqueness_score(synthetic_data)
        
        # Overall aggregated scores with financial weighting
        if column_scores:
            aggregated = self._aggregate_column_scores(column_scores)
            results.update(aggregated)
            
            # Create summary of individual vs overall performance
            results["_performance_summary"] = self._create_performance_summary(column_scores, aggregated)
        
        return results
    
    def _dict_to_column_matrices(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Convert dictionary of time series DataFrames to per-column matrix representation.
        
        This method creates separate matrices for each column, preventing scale domination
        issues that occur when flattening all features together. Each column is processed
        independently with appropriate normalization.
        
        Args:
            data_dict (Dict[str, pd.DataFrame]): Dictionary of time series DataFrames
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping column names to matrices
                                  where each row is a time series for that column
                                  Shape per matrix: (n_series, min_length)
        
        Note:
            - Processes each numeric column separately to avoid scale bias
            - Excludes OpenInt column (all zeros)
            - Uses minimum length across all series for normalization
            - Only includes overlapping timesteps, no padding applied
        """
        if not data_dict:
            return {}
        
        # Get all numeric columns, excluding OpenInt
        all_columns = set()
        for df in data_dict.values():
            if not df.empty:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                # Filter out OpenInt column
                filtered_cols = [col for col in numeric_cols if col.lower() != 'openint']
                all_columns.update(filtered_cols)
        
        if not all_columns:
            return {}
        
        # Find minimum length across all series
        min_length = float('inf')
        for df in data_dict.values():
            if not df.empty:
                min_length = min(min_length, len(df))
        
        if min_length == float('inf'):
            return {}
        
        # Create matrix for each column using only overlapping timesteps (no padding)
        column_matrices = {}
        for column in all_columns:
            column_data = []
            
            for series_id, df in data_dict.items():
                if df.empty or column not in df.columns:
                    continue
                    
                series_data = df[column].values
                
                # Use only overlapping timesteps - truncate to min_length, no padding
                if len(series_data) >= min_length:
                    normalized_series = series_data[:min_length]
                    column_data.append(normalized_series)
                # Skip series that are shorter than min_length - no padding applied
                
            if column_data:
                column_matrices[column] = np.array(column_data)
        
        return column_matrices
    
    def _dict_to_normalized_column_matrices(self, original_data: Dict[str, pd.DataFrame], 
                                          synthetic_data: Dict[str, pd.DataFrame]) -> tuple:
        """
        Convert data to normalized column matrices using original data statistics only.
        
        This method computes normalization statistics from ORIGINAL data only
        to ensure fair evaluation without any synthetic data influence on scaling.
        
        Args:
            original_data: Original time series data
            synthetic_data: Synthetic time series data
            
        Returns:
            tuple: (orig_columns, synth_columns, normalization_stats)
        """
        # First get raw column matrices
        orig_raw = self._dict_to_column_matrices(original_data)
        synth_raw = self._dict_to_column_matrices(synthetic_data)
        
        if not orig_raw or not synth_raw:
            return {}, {}, {}
        
        # Compute normalization statistics from ORIGINAL data only
        normalization_stats = {}
        orig_normalized = {}
        synth_normalized = {}
        
        common_columns = set(orig_raw.keys()) & set(synth_raw.keys())
        
        for column in common_columns:
            orig_col = orig_raw[column]
            synth_col = synth_raw[column]
            
            # Use ONLY original data for normalization statistics
            orig_flat = orig_col.flatten()
            
            # Compute statistics from original data only
            orig_mean = np.mean(orig_flat)
            orig_std = np.std(orig_flat, ddof=1)
            orig_min = np.min(orig_flat)
            orig_max = np.max(orig_flat)
            
            # Store normalization stats
            normalization_stats[column] = {
                "mean": orig_mean,
                "std": orig_std,
                "min": orig_min,
                "max": orig_max,
                "range": orig_max - orig_min,
                "n_samples": len(orig_flat)
            }
            
            # Apply z-score normalization using original data statistics
            if orig_std > 1e-8:  # Avoid division by zero
                orig_normalized[column] = (orig_col - orig_mean) / orig_std
                synth_normalized[column] = (synth_col - orig_mean) / orig_std
            else:
                # Fallback for constant columns
                orig_normalized[column] = orig_col - orig_mean
                synth_normalized[column] = synth_col - orig_mean
        
        return orig_normalized, synth_normalized, normalization_stats
    
    def _create_performance_summary(self, column_scores: Dict[str, Dict[str, float]], 
                                  aggregated: Dict[str, float]) -> Dict[str, any]:
        """Create summary comparing individual column vs overall performance."""
        summary = {
            "best_column": None,
            "worst_column": None,
            "column_rankings": [],
            "performance_gap": 0.0,
            "uniform_performance": True
        }
        
        # Calculate individual column overall scores
        col_overall_scores = {}
        for col, scores in column_scores.items():
            col_overall_scores[col] = np.mean([
                scores.get("variance_ratio", 0.0),
                scores.get("range_ratio", 0.0),
                scores.get("entropy_ratio", 0.0),
                scores.get("autocorr_diversity", 0.0),
                scores.get("coverage", 0.0)
            ])
        
        if col_overall_scores:
            # Find best and worst performing columns
            sorted_cols = sorted(col_overall_scores.items(), key=lambda x: x[1], reverse=True)
            summary["best_column"] = {"name": sorted_cols[0][0], "score": sorted_cols[0][1]}
            summary["worst_column"] = {"name": sorted_cols[-1][0], "score": sorted_cols[-1][1]}
            summary["column_rankings"] = sorted_cols
            
            # Calculate performance gap
            best_score = sorted_cols[0][1]
            worst_score = sorted_cols[-1][1]
            summary["performance_gap"] = best_score - worst_score
            
            # Check if performance is relatively uniform (gap < 0.2)
            summary["uniform_performance"] = summary["performance_gap"] < 0.2
            
            # Overall vs individual comparison
            overall_score = aggregated.get("overall_diversity_score", 0.0)
            summary["overall_vs_best"] = overall_score - best_score
            summary["overall_vs_worst"] = overall_score - worst_score
        
        return summary
    
    def _compute_column_diversity(self, orig_matrix: np.ndarray, synth_matrix: np.ndarray, column: str) -> Dict[str, float]:
        """
        Compute diversity metrics for a single column.
        
        Args:
            orig_matrix (np.ndarray): Original data matrix for this column
            synth_matrix (np.ndarray): Synthetic data matrix for this column  
            column (str): Column name for context
            
        Returns:
            Dict[str, float]: Column-specific diversity metrics
        """
        results = {}
        
        try:
            # Variance ratio
            orig_var = np.var(orig_matrix, ddof=1) if len(orig_matrix) > 1 else 0.0
            synth_var = np.var(synth_matrix, ddof=1) if len(synth_matrix) > 1 else 0.0
            if orig_var > 0:
                results["variance_ratio"] = min(synth_var / orig_var, 1.0)
            else:
                results["variance_ratio"] = 1.0 if synth_var == 0 else 0.0
                
            # Range ratio
            orig_range = np.ptp(orig_matrix)
            synth_range = np.ptp(synth_matrix) 
            if orig_range > 0:
                results["range_ratio"] = min(synth_range / orig_range, 1.0)
            else:
                results["range_ratio"] = 1.0 if synth_range == 0 else 0.0
                
            # Entropy ratio
            orig_entropy = self._calculate_entropy(orig_matrix.flatten())
            synth_entropy = self._calculate_entropy(synth_matrix.flatten())
            if orig_entropy > 0:
                results["entropy_ratio"] = min(synth_entropy / orig_entropy, 1.0) 
            else:
                results["entropy_ratio"] = 1.0 if synth_entropy == 0 else 0.0
                
            # Autocorrelation diversity (for time series columns)
            results["autocorr_diversity"] = self._column_autocorr_diversity(orig_matrix, synth_matrix)
            
            # Column-specific coverage
            results["coverage"] = self._column_coverage(orig_matrix, synth_matrix)
            
        except Exception as e:
            print(f"Warning: Error computing diversity for column {column}: {e}")
            # Fallback values
            for metric in ["variance_ratio", "range_ratio", "entropy_ratio", "autocorr_diversity", "coverage"]:
                results[metric] = 0.0
                
        return results
    
    def _column_autocorr_diversity(self, orig_matrix: np.ndarray, synth_matrix: np.ndarray) -> float:
        """Compute autocorrelation diversity for a single column."""
        try:
            orig_autocorrs = []
            synth_autocorrs = []
            
            # Compute lag-1 autocorrelation for each series
            for series in orig_matrix:
                if len(series) > 1:
                    # Remove any constant series that would cause correlation issues
                    if np.std(series) > 1e-8:
                        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                        if not np.isnan(autocorr) and not np.isinf(autocorr):
                            orig_autocorrs.append(autocorr)
                        
            for series in synth_matrix:
                if len(series) > 1:
                    # Remove any constant series that would cause correlation issues
                    if np.std(series) > 1e-8:
                        autocorr = np.corrcoef(series[:-1], series[1:])[0, 1]
                        if not np.isnan(autocorr) and not np.isinf(autocorr):
                            synth_autocorrs.append(autocorr)
            
            # If we have no valid autocorrelations, try alternative method
            if not orig_autocorrs or not synth_autocorrs:
                # Fallback: compute autocorr on flattened data if individual series fail
                if len(orig_matrix.flatten()) > 1 and len(synth_matrix.flatten()) > 1:
                    orig_flat = orig_matrix.flatten()
                    synth_flat = synth_matrix.flatten()
                    
                    if np.std(orig_flat) > 1e-8 and np.std(synth_flat) > 1e-8:
                        orig_autocorr = np.corrcoef(orig_flat[:-1], orig_flat[1:])[0, 1]
                        synth_autocorr = np.corrcoef(synth_flat[:-1], synth_flat[1:])[0, 1]
                        
                        if not (np.isnan(orig_autocorr) or np.isnan(synth_autocorr)):
                            diff = abs(orig_autocorr - synth_autocorr)
                            return max(0.0, 1.0 - diff)
                
                return 0.0
                
            # Compare distributions of autocorrelations
            orig_mean = np.mean(orig_autocorrs)
            synth_mean = np.mean(synth_autocorrs)
            diff = abs(orig_mean - synth_mean)
            
            # Use a more reasonable comparison - don't penalize too heavily for small differences
            similarity = max(0.0, 1.0 - min(diff, 1.0))
            return similarity
            
        except Exception as e:
            # For debugging - this should help identify the issue
            import traceback
            print(f"Autocorr calculation error: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return 0.0
    
    def _column_coverage(self, orig_matrix: np.ndarray, synth_matrix: np.ndarray) -> float:
        """Compute coverage ratio for a single column."""
        try:
            orig_flat = orig_matrix.flatten()
            synth_flat = synth_matrix.flatten()
            
            orig_min, orig_max = np.min(orig_flat), np.max(orig_flat)
            synth_min, synth_max = np.min(synth_flat), np.max(synth_flat)
            
            if orig_max == orig_min:
                return 1.0 if synth_max == synth_min else 0.0
                
            # Calculate what fraction of original range is covered by synthetic
            overlap_min = max(orig_min, synth_min)
            overlap_max = min(orig_max, synth_max)
            
            if overlap_max <= overlap_min:
                return 0.0
                
            coverage = (overlap_max - overlap_min) / (orig_max - orig_min)
            return min(coverage, 1.0)
            
        except Exception:
            return 0.0
    
    def _aggregate_column_scores(self, column_scores: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Aggregate column-specific scores with financial domain weighting.
        
        Args:
            column_scores: Dictionary mapping column names to their metric scores
            
        Returns:
            Dict[str, float]: Aggregated scores for backward compatibility
        """
        results = {}
        
        # Define financial weights - Price features get higher priority
        available_cols = list(column_scores.keys())
        price_cols = [col for col in available_cols if col in ['Open', 'High', 'Low', 'Close']]
        has_volume = 'Volume' in available_cols
        
        if price_cols and has_volume:
            # Both price and volume available - use 70/30 split
            price_weight = 0.7 / len(price_cols) if price_cols else 0
            volume_weight = 0.3
        elif price_cols:
            # Only price features available
            price_weight = 1.0 / len(price_cols)
            volume_weight = 0
        elif has_volume:
            # Only volume available
            price_weight = 0
            volume_weight = 1.0
        else:
            # Equal weighting fallback
            price_weight = volume_weight = 1.0 / len(available_cols) if available_cols else 0
        
        # Aggregate each metric type
        metrics = ['variance_ratio', 'range_ratio', 'entropy_ratio', 'autocorr_diversity', 'coverage']
        
        for metric in metrics:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for col, scores in column_scores.items():
                if metric in scores:
                    weight = price_weight if col in price_cols else volume_weight
                    weighted_sum += scores[metric] * weight
                    total_weight += weight
            
            if total_weight > 0:
                results[metric] = weighted_sum / total_weight
            else:
                results[metric] = 0.0
        
        # Overall diversity score
        diversity_components = [results.get(m, 0.0) for m in metrics]
        diversity_components.extend([
            results.get('coverage_ratio', 0.0), 
            results.get('uniqueness_score', 0.0)
        ])
        
        results["overall_diversity_score"] = np.mean([x for x in diversity_components if x is not None])
        
        return results
    
    def _compute_legacy_diversity(self, original_data: Dict[str, pd.DataFrame], 
                                 synthetic_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Fallback to original flattened method if needed."""
        results = {}
        
        results["coverage_ratio"] = self.coverage_ratio(original_data, synthetic_data)
        results["uniqueness_score"] = self.uniqueness_score(synthetic_data)
        
        statistical_div = self.statistical_diversity(original_data, synthetic_data)
        results.update(statistical_div)
        
        temporal_div = self.temporal_pattern_diversity(original_data, synthetic_data)
        results.update(temporal_div)
        
        diversity_score = np.mean([
            results["coverage_ratio"],
            results["uniqueness_score"], 
            results["variance_ratio"],
            results["range_ratio"],
            results["entropy_ratio"],
            results["autocorr_diversity"],
            results["trend_diversity"]
        ])
        
        results["overall_diversity_score"] = diversity_score
        
        return results
    
    def _dict_to_matrix(self, data_dict: Dict[str, pd.DataFrame], target_length: int = None) -> np.ndarray:
        """Convert dictionary of dataframes to matrix using fair overlapping data only."""
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
    
    def _calculate_entropy(self, data: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Shannon entropy of data using histogram-based discretization.
        
        Args:
            data (np.ndarray): Input data array
            bins (int): Number of histogram bins for discretization
        
        Returns:
            float: Shannon entropy value (higher = more diverse)
        
        Note:
            - Adds small epsilon to avoid log(0) issues
            - Returns 0.0 for empty or invalid data
        """
        if len(data) == 0:
            return 0.0
        
        try:
            hist, _ = np.histogram(data, bins=bins, density=True)
            hist = hist + 1e-10
            return entropy(hist)
        except:
            return 0.0
    
    def _calculate_autocorrelation(self, df: pd.DataFrame, max_lag: int = 10) -> np.ndarray:
        """
        Calculate autocorrelation features for temporal pattern analysis.
        
        Args:
            df (pd.DataFrame): Time series DataFrame
            max_lag (int): Maximum lag for autocorrelation computation
        
        Returns:
            np.ndarray: Array of autocorrelation values across all numeric columns
        
        Note:
            - Handles missing values and insufficient data gracefully
            - Pads with zeros when series too short for specified lags
        """
        autocorrs = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > max_lag:
                col_autocorrs = []
                for lag in range(1, min(max_lag + 1, len(series))):
                    autocorr = series.autocorr(lag=lag)
                    if pd.isna(autocorr):
                        autocorr = 0.0
                    col_autocorrs.append(autocorr)
                
                while len(col_autocorrs) < max_lag:
                    col_autocorrs.append(0.0)
                
                autocorrs.extend(col_autocorrs)
            else:
                autocorrs.extend([0.0] * max_lag)
        
        return np.array(autocorrs) if autocorrs else np.array([0.0])
    
    def _calculate_trend_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract trend characteristics from time series data.
        
        Args:
            df (pd.DataFrame): Time series DataFrame
        
        Returns:
            np.ndarray: Array containing slope and second derivative features
        
        Note:
            - Uses linear regression for slope estimation
            - Second derivative captures curvature/acceleration
            - Returns zeros for insufficient data points
        """
        trends = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) > 1:
                x = np.arange(len(series))
                slope = np.polyfit(x, series, 1)[0]
                trends.append(slope)
                
                second_derivative = np.mean(np.diff(series, n=2)) if len(series) > 2 else 0.0
                trends.append(second_derivative)
            else:
                trends.extend([0.0, 0.0])
        
        return np.array(trends) if trends else np.array([0.0])

if __name__ == "__main__":
    from data_loader import TimeSeriesDataLoader
    
    loader = TimeSeriesDataLoader("TimeSeries")
    diversity_evaluator = DiversityMetrics()
    
    print("Testing Diversity Metrics...")
    
    _, original_cond = loader.load_conditional_data("original")
    _, synthetic_cond = loader.load_conditional_data("tsv2")
    
    if original_cond and synthetic_cond:
        diversity_results = diversity_evaluator.compute_all_diversity_metrics(
            original_cond, synthetic_cond
        )
        
        print("\nConditional Generation Diversity Results:")
        for metric, value in diversity_results.items():
            print(f"  {metric}: {value:.4f}")
    
    original_uncond = loader.load_unconditional_data("original")
    synthetic_uncond = loader.load_unconditional_data("tsv2")
    
    if original_uncond and synthetic_uncond:
        diversity_results = diversity_evaluator.compute_all_diversity_metrics(
            original_uncond, synthetic_uncond
        )
        
        print("\nUnconditional Generation Diversity Results:")
        for metric, value in diversity_results.items():
            print(f"  {metric}: {value:.4f}")