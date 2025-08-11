"""
Time Series Evaluation Pipeline Module

This module orchestrates comprehensive evaluation of synthetic time series data
across four key dimensions: Diversity, Fidelity, Privacy, and Utility. It provides
a complete evaluation framework based on TSGBench methodology and industry best practices.

The pipeline automates the entire evaluation process from data loading to report generation,
making it suitable for production use in synthetic data validation workflows.

Key Features:
- Automated evaluation across all four metric categories
- Support for both conditional and unconditional generation
- Weighted scoring system based on research recommendations
- Comprehensive error handling and validation
- JSON result export with detailed metadata
- Progress tracking and timing information

References:
- Ang, Yihao, et al. "TSGBench: Time Series Generation Benchmark." VLDB 2023
- Xu et al. "Modeling Tabular Data using Conditional GAN." NeurIPS 2019
- Jordon et al. "PATE-GAN: Generating Synthetic Data with Differential Privacy." ICLR 2019

"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_loader import TimeSeriesDataLoader
from diversity_metrics import DiversityMetrics
from fidelity_metrics import FidelityMetrics
from privacy_metrics import PrivacyMetrics
from utility_metrics import UtilityMetrics

class TimeSeriesEvaluationPipeline:
    """
    Comprehensive evaluation pipeline for synthetic time series data.
    
    This class orchestrates the complete evaluation workflow, coordinating
    data loading, metric computation across four evaluation dimensions,
    and result aggregation with business-friendly scoring.
    
    Attributes:
        data_loader (TimeSeriesDataLoader): Handles data loading operations
        diversity_evaluator (DiversityMetrics): Computes diversity metrics
        fidelity_evaluator (FidelityMetrics): Computes fidelity metrics
        privacy_evaluator (PrivacyMetrics): Computes privacy risk metrics
        utility_evaluator (UtilityMetrics): Computes utility metrics
        evaluation_results (Dict): Stores all evaluation results
        evaluation_metadata (Dict): Stores evaluation metadata
    
    Example:
        >>> pipeline = TimeSeriesEvaluationPipeline("TimeSeries")
        >>> results = pipeline.evaluate_all_variants()
        >>> print(f"Best variant: {pipeline.get_best_variant(results)}")
    """
    def __init__(self, data_root: str = "TimeSeries"):
        """
        Initialize the evaluation pipeline with data source and metric evaluators.
        
        Args:
            data_root (str): Path to the root directory containing TimeSeries data
                           Expected to have 'conditional generation' and 
                           'unconditional generation' subdirectories
        
        Example:
            >>> pipeline = TimeSeriesEvaluationPipeline("data/TimeSeries")
            >>> # Pipeline ready for evaluation
        """
        self.data_loader = TimeSeriesDataLoader(data_root)
        self.diversity_evaluator = DiversityMetrics()
        self.fidelity_evaluator = FidelityMetrics()
        self.privacy_evaluator = PrivacyMetrics()
        self.utility_evaluator = UtilityMetrics()
        
        self.evaluation_results = {}
        self.evaluation_metadata = {}
    
    def evaluate_synthetic_variant(self, variant: str, data_type: str = "conditional") -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"Evaluating {data_type} {variant}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            if data_type == "conditional":
                _, original_data = self.data_loader.load_conditional_data("original")
                _, synthetic_data = self.data_loader.load_conditional_data(variant)
            else:
                original_data = self.data_loader.load_unconditional_data("original")
                synthetic_data = self.data_loader.load_unconditional_data(variant)
            
            if not original_data or not synthetic_data:
                print(f"Warning: No data found for {variant}")
                return self._create_empty_results()
            
            results = {}
            
            print("Computing Diversity Metrics...")
            diversity_results = self.diversity_evaluator.compute_all_diversity_metrics(
                original_data, synthetic_data
            )
            results["diversity"] = diversity_results
            
            print("Computing Fidelity Metrics...")
            fidelity_results = self.fidelity_evaluator.compute_all_fidelity_metrics(
                original_data, synthetic_data
            )
            results["fidelity"] = fidelity_results
            
            print("Computing Privacy Metrics...")
            privacy_results = self.privacy_evaluator.compute_all_privacy_metrics(
                original_data, synthetic_data
            )
            results["privacy"] = privacy_results
            
            print("Computing Utility Metrics...")
            utility_results = self.utility_evaluator.compute_all_utility_metrics(
                original_data, synthetic_data
            )
            results["utility"] = utility_results
            
            print("Computing Column-wise Metrics...")
            # Add per-column fidelity, privacy, utility metrics
            column_fidelity = self.fidelity_evaluator.compute_column_fidelity_metrics(
                original_data, synthetic_data
            )
            results["column_fidelity"] = column_fidelity
            
            column_privacy = self.privacy_evaluator.compute_column_privacy_metrics(
                original_data, synthetic_data
            )
            results["column_privacy"] = column_privacy
            
            column_utility = self.utility_evaluator.compute_column_utility_metrics(
                original_data, synthetic_data
            )
            results["column_utility"] = column_utility
            
            overall_score = self._compute_overall_score(results)
            results["overall_score"] = overall_score
            
            evaluation_time = time.time() - start_time
            
            metadata = {
                "variant": variant,
                "data_type": data_type,
                "evaluation_time": evaluation_time,
                "timestamp": datetime.now().isoformat(),
                "original_samples": len(original_data),
                "synthetic_samples": len(synthetic_data)
            }
            
            results["metadata"] = metadata
            
            print(f"Evaluation completed in {evaluation_time:.2f} seconds")
            print(f"Overall Score: {overall_score:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error evaluating {variant}: {str(e)}")
            return self._create_empty_results()
    
    def evaluate_all_variants(self) -> Dict[str, Any]:
        print("Starting comprehensive evaluation of all synthetic variants...")
        print(f"Timestamp: {datetime.now().isoformat()}")
        
        all_results = {
            "conditional": {},
            "unconditional": {},
            "baselines": {},
            "summary": {},
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "framework_version": "1.0.0"
            }
        }
        
        conditional_variants = self.data_loader.get_available_variants("conditional")
        unconditional_variants = self.data_loader.get_available_variants("unconditional")
        
        # Separate baselines from true synthetic variants
        baseline_variants = ["original_noise"]
        synthetic_conditional = [v for v in conditional_variants if v not in ["original"] + baseline_variants]
        synthetic_unconditional = [v for v in unconditional_variants if v != "original"]
        
        # Evaluate baseline variants
        for variant in baseline_variants:
            if variant in conditional_variants:
                print(f"\nProcessing baseline: {variant}")
                try:
                    results = self.evaluate_synthetic_variant(variant, "conditional")
                    results["metadata"]["variant_type"] = "baseline"
                    results["metadata"]["description"] = "Original data with added noise (baseline comparison)"
                    all_results["baselines"][variant] = results
                except Exception as e:
                    print(f"Failed to evaluate baseline {variant}: {str(e)}")
                    all_results["baselines"][variant] = self._create_empty_results()
        
        # Evaluate true synthetic variants
        for variant in synthetic_conditional:
            print(f"\nProcessing synthetic conditional variant: {variant}")
            try:
                results = self.evaluate_synthetic_variant(variant, "conditional")
                results["metadata"]["variant_type"] = "synthetic"
                all_results["conditional"][variant] = results
            except Exception as e:
                print(f"Failed to evaluate {variant}: {str(e)}")
                all_results["conditional"][variant] = self._create_empty_results()
        
        for variant in synthetic_unconditional:
            print(f"\nProcessing synthetic unconditional variant: {variant}")
            try:
                results = self.evaluate_synthetic_variant(variant, "unconditional")
                results["metadata"]["variant_type"] = "synthetic"
                all_results["unconditional"][variant] = results
            except Exception as e:
                print(f"Failed to evaluate {variant}: {str(e)}")
                all_results["unconditional"][variant] = self._create_empty_results()
        
        all_results["summary"] = self._generate_summary(all_results)
        
        return all_results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        try:
            diversity_score = results["diversity"].get("overall_diversity_score", 0.0)
            fidelity_score = results["fidelity"].get("overall_fidelity_score", 0.0)
            privacy_score = results["privacy"].get("overall_privacy_score", 0.0)
            utility_score = results["utility"].get("overall_utility_score", 0.0)
            
            if np.isnan(diversity_score) or np.isinf(diversity_score):
                diversity_score = 0.0
            if np.isnan(fidelity_score) or np.isinf(fidelity_score):
                fidelity_score = 0.0
            if np.isnan(privacy_score) or np.isinf(privacy_score):
                privacy_score = 0.0
            if np.isnan(utility_score) or np.isinf(utility_score):
                utility_score = 0.0
            
            weights = {
                "diversity": 0.25,
                "fidelity": 0.35,  # Increased (core TSGBench metrics)
                "privacy": 0.15,   # Decreased (simplified approach)
                "utility": 0.25
            }
            
            weighted_components = [
                weights["diversity"] * diversity_score,
                weights["fidelity"] * fidelity_score,
                weights["privacy"] * privacy_score,
                weights["utility"] * utility_score
            ]
            
            # Filter out any remaining NaN values
            valid_components = [x for x in weighted_components if not (np.isnan(x) or np.isinf(x))]
            
            if len(valid_components) > 0:
                overall_score = sum(valid_components)
            else:
                overall_score = 0.0
                
            # Final NaN check and bounds
            if np.isnan(overall_score) or np.isinf(overall_score):
                overall_score = 0.0
            else:
                overall_score = max(0.0, min(1.0, overall_score))  # Clamp to [0,1]
            
            return overall_score
        except Exception as e:
            print(f"Error computing overall score: {str(e)}")
            return 0.0
    
    def _create_empty_results(self) -> Dict[str, Any]:
        return {
            "diversity": {"overall_diversity_score": 0.0},
            "fidelity": {"overall_fidelity_score": 0.0},
            "privacy": {"overall_privacy_score": 0.0},
            "utility": {"overall_utility_score": 0.0},
            "overall_score": 0.0,
            "metadata": {
                "error": True,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _generate_summary(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            "best_performers": {},
            "baseline_performance": {},
            "metric_averages": {},
            "recommendations": []
        }
        
        try:
            conditional_scores = {}
            unconditional_scores = {}
            baseline_scores = {}
            
            # Collect synthetic variant scores (excluding baselines) with NaN filtering
            for variant, results in all_results["conditional"].items():
                if "overall_score" in results:
                    score = results["overall_score"]
                    if not (np.isnan(score) or np.isinf(score)):
                        conditional_scores[variant] = score
            
            for variant, results in all_results["unconditional"].items():
                if "overall_score" in results:
                    score = results["overall_score"]
                    if not (np.isnan(score) or np.isinf(score)):
                        unconditional_scores[variant] = score
            
            # Collect baseline scores separately
            if "baselines" in all_results:
                for variant, results in all_results["baselines"].items():
                    if "overall_score" in results:
                        baseline_scores[variant] = results["overall_score"]
            
            if conditional_scores:
                best_conditional = max(conditional_scores.items(), key=lambda x: x[1])
                summary["best_performers"]["conditional"] = {
                    "variant": best_conditional[0],
                    "score": best_conditional[1]
                }
            
            if unconditional_scores:
                best_unconditional = max(unconditional_scores.items(), key=lambda x: x[1])
                summary["best_performers"]["unconditional"] = {
                    "variant": best_unconditional[0],
                    "score": best_unconditional[1]
                }
            
            # Add baseline performance for comparison
            if baseline_scores:
                summary["baseline_performance"] = {
                    variant: {"score": score, "description": "Original data with noise"} 
                    for variant, score in baseline_scores.items()
                }
            
            summary["metric_averages"] = self._compute_separate_metric_averages(all_results)
            summary["recommendations"] = self._generate_separate_recommendations(all_results)
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            summary["error"] = str(e)
        
        return summary
    
    def _compute_separate_metric_averages(self, all_results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Compute metric averages separately for conditional and unconditional generation"""
        averages = {
            "conditional": {"diversity": [], "fidelity": [], "privacy": [], "utility": [], "overall": []},
            "unconditional": {"diversity": [], "fidelity": [], "privacy": [], "utility": [], "overall": []}
        }
        
        # Conditional generation averages (exclude baselines)
        if "conditional" in all_results:
            for variant, results in all_results["conditional"].items():
                if isinstance(results, dict) and "overall_score" in results:
                    averages["conditional"]["diversity"].append(results.get("diversity", {}).get("overall_diversity_score", 0.0))
                    averages["conditional"]["fidelity"].append(results.get("fidelity", {}).get("overall_fidelity_score", 0.0))
                    averages["conditional"]["privacy"].append(results.get("privacy", {}).get("overall_privacy_score", 0.0))
                    averages["conditional"]["utility"].append(results.get("utility", {}).get("overall_utility_score", 0.0))
                    averages["conditional"]["overall"].append(results["overall_score"])
        
        # Unconditional generation averages
        if "unconditional" in all_results:
            for variant, results in all_results["unconditional"].items():
                if isinstance(results, dict) and "overall_score" in results:
                    averages["unconditional"]["diversity"].append(results.get("diversity", {}).get("overall_diversity_score", 0.0))
                    averages["unconditional"]["fidelity"].append(results.get("fidelity", {}).get("overall_fidelity_score", 0.0))
                    averages["unconditional"]["privacy"].append(results.get("privacy", {}).get("overall_privacy_score", 0.0))
                    averages["unconditional"]["utility"].append(results.get("utility", {}).get("overall_utility_score", 0.0))
                    averages["unconditional"]["overall"].append(results["overall_score"])
        
        # Calculate averages with NaN handling
        result = {}
        for gen_type, metrics in averages.items():
            result[gen_type] = {}
            for metric, values in metrics.items():
                if values:
                    # Filter out NaN values
                    valid_values = [v for v in values if not (np.isnan(v) or np.isinf(v))]
                    if valid_values:
                        avg = sum(valid_values) / len(valid_values)
                        # Final NaN check
                        result[gen_type][metric] = avg if not (np.isnan(avg) or np.isinf(avg)) else 0.0
                    else:
                        result[gen_type][metric] = 0.0
                else:
                    result[gen_type][metric] = 0.0
        
        return result
    
    def _generate_separate_recommendations(self, all_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate recommendations separately for conditional and unconditional generation"""
        recommendations = {
            "conditional": [],
            "unconditional": [],
            "baseline_comparison": []
        }
        
        try:
            averages = self._compute_separate_metric_averages(all_results)
            
            # Conditional generation recommendations
            if averages["conditional"]["overall"] > 0:
                cond_avg = averages["conditional"]
                if cond_avg["diversity"] < 0.5:
                    recommendations["conditional"].append("LOW DIVERSITY: Conditional synthetic data lacks sufficient variety")
                if cond_avg["fidelity"] < 0.6:
                    recommendations["conditional"].append("LOW FIDELITY: Statistical properties don't match original conditional data")
                if cond_avg["privacy"] < 0.4:
                    recommendations["conditional"].append("PRIVACY RISK: High vulnerability to membership inference attacks")
                if cond_avg["utility"] < 0.5:
                    recommendations["conditional"].append("LIMITED UTILITY: Poor performance for conditional prediction tasks")
                
                if cond_avg["overall"] > 0.7:
                    recommendations["conditional"].append("EXCELLENT: Conditional generation ready for production")
                elif cond_avg["overall"] > 0.5:
                    recommendations["conditional"].append("MODERATE: Conditional generation needs refinement")
                else:
                    recommendations["conditional"].append("POOR: Conditional generation requires major improvements")
            
            # Unconditional generation recommendations
            if averages["unconditional"]["overall"] > 0:
                uncond_avg = averages["unconditional"]
                if uncond_avg["diversity"] < 0.5:
                    recommendations["unconditional"].append("LOW DIVERSITY: Unconditional synthetic data lacks variety")
                if uncond_avg["fidelity"] < 0.6:
                    recommendations["unconditional"].append("LOW FIDELITY: Statistical properties don't match original data")
                if uncond_avg["privacy"] < 0.4:
                    recommendations["unconditional"].append("PRIVACY RISK: High vulnerability to privacy attacks")
                if uncond_avg["utility"] < 0.5:
                    recommendations["unconditional"].append("LIMITED UTILITY: Poor performance for general tasks")
                
                if uncond_avg["overall"] > 0.7:
                    recommendations["unconditional"].append("EXCELLENT: Unconditional generation ready for production")
                elif uncond_avg["overall"] > 0.5:
                    recommendations["unconditional"].append("MODERATE: Unconditional generation needs refinement")
                else:
                    recommendations["unconditional"].append("POOR: Unconditional generation requires major improvements")
            
            # Baseline comparison recommendations
            if "baselines" in all_results:
                baseline_scores = {variant: results["overall_score"] for variant, results in all_results["baselines"].items() if "overall_score" in results}
                if baseline_scores:
                    max_baseline = max(baseline_scores.values())
                    if averages["conditional"]["overall"] > 0 and averages["conditional"]["overall"] < max_baseline:
                        recommendations["baseline_comparison"].append(f"UNDERPERFORMING: Conditional synthetic methods score lower than baseline (max: {max_baseline:.3f})")
                    if averages["unconditional"]["overall"] > 0 and averages["unconditional"]["overall"] < max_baseline:
                        recommendations["baseline_comparison"].append(f"UNDERPERFORMING: Unconditional synthetic methods score lower than baseline (max: {max_baseline:.3f})")
                        
        except Exception as e:
            recommendations["conditional"].append(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def save_results(self, results: Dict[str, Any], output_file: str = None) -> str:
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"evaluation_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results saved to: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return ""
    
    def load_results(self, input_file: str) -> Dict[str, Any]:
        try:
            with open(input_file, 'r') as f:
                results = json.load(f)
            print(f"Results loaded from: {input_file}")
            return results
        except Exception as e:
            print(f"Error loading results: {str(e)}")
            return {}
    
    def print_summary_report(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("SYNTHETIC TIME SERIES EVALUATION REPORT")
        print("="*80)
        
        if "evaluation_metadata" in results:
            print(f"Generated: {results['evaluation_metadata']['timestamp']}")
            print(f"Framework Version: {results['evaluation_metadata']['framework_version']}")
        
        print("\nBEST SYNTHETIC PERFORMERS:")
        if "summary" in results and "best_performers" in results["summary"]:
            for data_type, best in results["summary"]["best_performers"].items():
                print(f"  {data_type.upper()}: {best['variant']} (Score: {best['score']:.4f})")
        
        print("\nBASELINE PERFORMANCE:")
        if "summary" in results and "baseline_performance" in results["summary"]:
            for variant, info in results["summary"]["baseline_performance"].items():
                print(f"  {variant}: {info['score']:.4f} ({info['description']})")
        
        print("\nMETRIC AVERAGES:")
        if "summary" in results and "metric_averages" in results["summary"]:
            metric_averages = results["summary"]["metric_averages"]
            
            if "conditional" in metric_averages:
                print("  CONDITIONAL GENERATION:")
                for metric, avg in metric_averages["conditional"].items():
                    print(f"    {metric.upper()}: {avg:.4f}")
            
            if "unconditional" in metric_averages:
                print("  UNCONDITIONAL GENERATION:")
                for metric, avg in metric_averages["unconditional"].items():
                    print(f"    {metric.upper()}: {avg:.4f}")
        
        print("\nRECOMMENDATIONS:")
        if "summary" in results and "recommendations" in results["summary"]:
            recommendations = results["summary"]["recommendations"]
            
            if "conditional" in recommendations and recommendations["conditional"]:
                print("  CONDITIONAL GENERATION:")
                for i, rec in enumerate(recommendations["conditional"], 1):
                    print(f"    {i}. {rec}")
                    
            if "unconditional" in recommendations and recommendations["unconditional"]:
                print("  UNCONDITIONAL GENERATION:")
                for i, rec in enumerate(recommendations["unconditional"], 1):
                    print(f"    {i}. {rec}")
                    
            if "baseline_comparison" in recommendations and recommendations["baseline_comparison"]:
                print("  BASELINE COMPARISON:")
                for i, rec in enumerate(recommendations["baseline_comparison"], 1):
                    print(f"    {i}. {rec}")
        
        print("\nDETAILED RESULTS:")
        
        # Show baselines first
        if "baselines" in results:
            print(f"\nBASELINE COMPARISON:")
            for variant, variant_results in results["baselines"].items():
                if isinstance(variant_results, dict) and "overall_score" in variant_results:
                    print(f"  {variant} (Original + Noise):")
                    print(f"    Overall Score: {variant_results['overall_score']:.4f}")
                    print(f"    Diversity: {variant_results.get('diversity', {}).get('overall_diversity_score', 0.0):.4f}")
                    print(f"    Fidelity: {variant_results.get('fidelity', {}).get('overall_fidelity_score', 0.0):.4f}")
                    print(f"    Privacy: {variant_results.get('privacy', {}).get('overall_privacy_score', 0.0):.4f}")
                    print(f"    Utility: {variant_results.get('utility', {}).get('overall_utility_score', 0.0):.4f}")
        
        # Show synthetic variants
        for data_type in ["conditional", "unconditional"]:
            if data_type in results:
                print(f"\n{data_type.upper()} SYNTHETIC GENERATION:")
                for variant, variant_results in results[data_type].items():
                    if isinstance(variant_results, dict) and "overall_score" in variant_results:
                        print(f"  {variant}:")
                        print(f"    Overall Score: {variant_results['overall_score']:.4f}")
                        print(f"    Diversity: {variant_results.get('diversity', {}).get('overall_diversity_score', 0.0):.4f}")
                        print(f"    Fidelity: {variant_results.get('fidelity', {}).get('overall_fidelity_score', 0.0):.4f}")
                        print(f"    Privacy: {variant_results.get('privacy', {}).get('overall_privacy_score', 0.0):.4f}")
                        print(f"    Utility: {variant_results.get('utility', {}).get('overall_utility_score', 0.0):.4f}")
        
        print("\n" + "="*80)

def main():
    pipeline = TimeSeriesEvaluationPipeline("TimeSeries")
    
    print("Time Series Synthetic Data Evaluation Pipeline")
    print("=" * 50)
    
    results = pipeline.evaluate_all_variants()
    
    output_file = pipeline.save_results(results)
    
    pipeline.print_summary_report(results)
    
    return results, output_file

if __name__ == "__main__":
    results, output_file = main()