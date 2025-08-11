import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class ReportGenerator:
    def __init__(self, style: str = "whitegrid"):
        plt.style.use('default')
        sns.set_style(style)
        self.colors = {
            "diversity": "#1f77b4",
            "fidelity": "#ff7f0e", 
            "privacy": "#2ca02c",
            "utility": "#d62728",
            "overall": "#9467bd",
            "conditional": "#2E86AB",      # Blue family for conditional
            "conditional_light": "#A23B72", # Darker blue-purple for conditional variants
            "unconditional": "#F18F01",    # Orange family for unconditional  
            "unconditional_light": "#C73E1D", # Red-orange for unconditional variants
            "baseline": "#28A745"           # Green for baseline (distinct from both families)
        }
        
    def generate_comprehensive_report(self, results: Dict[str, Any], 
                                    output_dir: str = "evaluation_report") -> str:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive evaluation report in: {output_path}")
        
        # Generate visualizations
        self._create_overall_performance_chart(results, output_path)
        self._create_metric_breakdown_charts(results, output_path)
        self._create_radar_charts(results, output_path)
        self._create_heatmap_comparison(results, output_path)
        self._create_time_series_comparison_plots(results, output_path)
        
        # Generate HTML report
        html_file = self._generate_html_report(results, output_path)
        
        # Business summary generation removed per user request
        
        print(f"Report generation completed. Main report: {html_file}")
        return str(html_file)
        
    def _create_overall_performance_chart(self, results: Dict[str, Any], output_path: Path):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Conditional generation performance (include baseline as first variant)
        if "conditional" in results or "baselines" in results:
            variants = []
            scores = []
            colors = []
            
            # Add baseline first if available (as part of conditional)
            if "baselines" in results:
                for variant, data in results["baselines"].items():
                    if isinstance(data, dict) and "overall_score" in data:
                        baseline_name = variant.replace("_", "+").title()  # Convert original_noise to Original+Noise
                        variants.append(baseline_name)
                        scores.append(data["overall_score"])
                        colors.append(self.colors["conditional"])  # Same color family as conditional
                        break
            
            # Add other conditional variants
            if "conditional" in results:
                color_variants = [self.colors["conditional"], self.colors["conditional_light"]]
                color_idx = 0
                for variant, data in results["conditional"].items():
                    if isinstance(data, dict) and "overall_score" in data:
                        variants.append(variant)
                        scores.append(data["overall_score"])
                        colors.append(color_variants[color_idx % len(color_variants)])
                        color_idx += 1
            
            if variants:
                bars1 = ax1.bar(variants, scores, color=colors, alpha=0.7, width=0.6)
                ax1.set_title("Conditional Generation", size=14, fontweight='bold')
                ax1.set_ylabel("Overall Score", size=12)
                ax1.set_ylim(0, 1.0)
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars1, scores):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Rotate x-axis labels for better readability
                ax1.tick_params(axis='x', rotation=45)
        
        # Unconditional generation performance (separate from baseline)
        if "unconditional" in results:
            variants = []
            scores = []
            colors = []
            
            # Add unconditional variants only
            color_variants = [self.colors["unconditional"], self.colors["unconditional_light"]]
            color_idx = 0
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict) and "overall_score" in data:
                    variants.append(variant)
                    scores.append(data["overall_score"])
                    colors.append(color_variants[color_idx % len(color_variants)])
                    color_idx += 1
            
            if variants:
                bars2 = ax2.bar(variants, scores, color=colors, alpha=0.7, width=0.6)
                ax2.set_title("Unconditional Generation", size=14, fontweight='bold')
                ax2.set_ylabel("Overall Score", size=12)
                ax2.set_ylim(0, 1.0)
                ax2.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars2, scores):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
                
                # Rotate x-axis labels for better readability
                ax2.tick_params(axis='x', rotation=45)
        
        # Combined comparison chart (all data together with proper spacing)
        all_variants = []
        all_scores = []
        all_colors = []
        all_labels = []
        positions = []
        pos_counter = 0
        
        # Add baseline and conditional variants first (left side)
        if "baselines" in results:
            for variant, data in results["baselines"].items():
                if isinstance(data, dict) and "overall_score" in data:
                    baseline_name = variant.replace("_", "+").title()
                    all_variants.append(baseline_name)
                    all_scores.append(data["overall_score"])
                    all_colors.append(self.colors["conditional"])
                    all_labels.append("Conditional")
                    positions.append(pos_counter)
                    pos_counter += 1
                    break
        
        if "conditional" in results:
            color_variants = [self.colors["conditional"], self.colors["conditional_light"]]
            color_idx = 0
            for variant, data in results["conditional"].items():
                if isinstance(data, dict) and "overall_score" in data:
                    all_variants.append(variant)
                    all_scores.append(data["overall_score"])
                    all_colors.append(color_variants[color_idx % len(color_variants)])
                    all_labels.append("Conditional")
                    positions.append(pos_counter)
                    pos_counter += 1
                    color_idx += 1
        
        # Add gap between conditional and unconditional
        pos_counter += 0.5
        
        # Add unconditional variants (right side)
        if "unconditional" in results:
            color_variants = [self.colors["unconditional"], self.colors["unconditional_light"]]
            color_idx = 0
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict) and "overall_score" in data:
                    all_variants.append(variant)
                    all_scores.append(data["overall_score"])
                    all_colors.append(color_variants[color_idx % len(color_variants)])
                    all_labels.append("Unconditional")
                    positions.append(pos_counter)
                    pos_counter += 1
                    color_idx += 1
        
        if all_variants:
            bars3 = ax3.bar(positions, all_scores, color=all_colors, alpha=0.7, width=0.6)
            ax3.set_title("Overall Performance Comparison", size=14, fontweight='bold')
            ax3.set_ylabel("Overall Score", size=12)
            ax3.set_ylim(0, 1.0)
            ax3.grid(axis='y', alpha=0.3)
            
            # Set custom x-axis labels
            ax3.set_xticks(positions)
            ax3.set_xticklabels(all_variants, rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars3, all_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # Create custom legend
            import matplotlib.patches as mpatches
            legend_elements = []
            if any(label == "Conditional" for label in all_labels):
                legend_elements.append(mpatches.Patch(color=self.colors["conditional"], label='Conditional'))
            if any(label == "Unconditional" for label in all_labels):
                legend_elements.append(mpatches.Patch(color=self.colors["unconditional"], label='Unconditional'))
            
            if legend_elements:
                ax3.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(output_path / "overall_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_metric_breakdown_charts(self, results: Dict[str, Any], output_path: Path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ["diversity", "fidelity", "privacy", "utility"]
        metric_labels = {
            "diversity": "overall_diversity_score",
            "fidelity": "overall_fidelity_score", 
            "privacy": "overall_privacy_score",
            "utility": "overall_utility_score"
        }
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Collect all unique variants and their scores
            variant_data = {}
            
            # Add baseline data
            if "baselines" in results:
                for variant, data in results["baselines"].items():
                    if isinstance(data, dict) and metric in data:
                        baseline_name = variant.replace("_", "+").title()
                        variant_data[baseline_name] = {
                            'conditional': data[metric].get(metric_labels[metric], 0.0),
                            'unconditional': 0.0
                        }
                        break
            
            # Add conditional data
            if "conditional" in results:
                for variant, data in results["conditional"].items():
                    if isinstance(data, dict) and metric in data:
                        if variant not in variant_data:
                            variant_data[variant] = {'conditional': 0.0, 'unconditional': 0.0}
                        variant_data[variant]['conditional'] = data[metric].get(metric_labels[metric], 0.0)
            
            # Add unconditional data
            if "unconditional" in results:
                for variant, data in results["unconditional"].items():
                    if isinstance(data, dict) and metric in data:
                        if variant not in variant_data:
                            variant_data[variant] = {'conditional': 0.0, 'unconditional': 0.0}
                        variant_data[variant]['unconditional'] = data[metric].get(metric_labels[metric], 0.0)
            
            if variant_data:
                # Extract data for plotting
                variants = list(variant_data.keys())
                conditional_scores = [variant_data[v]['conditional'] for v in variants]
                unconditional_scores = [variant_data[v]['unconditional'] for v in variants]
                
                # Create grouped bars
                x_pos = np.arange(len(variants))
                width = 0.35
                
                # Only show bars where scores > 0
                cond_bars = ax.bar(x_pos - width/2, conditional_scores, width, 
                                label='Conditional', color=self.colors["conditional"], alpha=0.8)
                uncond_bars = ax.bar(x_pos + width/2, unconditional_scores, width,
                                    label='Unconditional', color=self.colors["unconditional"], alpha=0.8)
                
                ax.set_title(f"{metric.upper()} Scores", size=14, fontweight='bold')
                ax.set_ylabel("Score", size=12)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(variants, rotation=45, ha='right')
                ax.set_ylim(0, 1.0)
                
                # Only show legend items that have data
                legend_elements = []
                if any(score > 0 for score in conditional_scores):
                    legend_elements.append(plt.Rectangle((0,0),1,1, color=self.colors["conditional"], label='Conditional'))
                if any(score > 0 for score in unconditional_scores):
                    legend_elements.append(plt.Rectangle((0,0),1,1, color=self.colors["unconditional"], label='Unconditional'))
                
                if legend_elements:
                    ax.legend(handles=legend_elements, loc='upper right')
                
                ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "metric_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_radar_charts(self, results: Dict[str, Any], output_path: Path):
        metrics = ["diversity", "fidelity", "privacy", "utility"]
        metric_labels = {
            "diversity": "overall_diversity_score",
            "fidelity": "overall_fidelity_score",
            "privacy": "overall_privacy_score", 
            "utility": "overall_utility_score"
        }
        
        # Create radar chart for each variant
        all_variants = {}
        
        # FIX: Add baseline (original noise) to radar chart
        if "baselines" in results:
            for variant, data in results["baselines"].items():
                if isinstance(data, dict):
                    scores = []
                    for metric in metrics:
                        if metric in data:
                            scores.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            scores.append(0.0)
                    baseline_display_name = variant.replace("_", "+").title()
                    all_variants[f"{baseline_display_name} (Baseline)"] = scores
        
        if "conditional" in results:
            for variant, data in results["conditional"].items():
                if isinstance(data, dict):
                    scores = []
                    for metric in metrics:
                        if metric in data:
                            scores.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            scores.append(0.0)
                    all_variants[f"{variant} (Conditional)"] = scores
        
        if "unconditional" in results:
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict):
                    scores = []
                    for metric in metrics:
                        if metric in data:
                            scores.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            scores.append(0.0)
                    all_variants[f"{variant} (Unconditional)"] = scores
        
        if all_variants:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            colors = plt.cm.Set1(np.linspace(0, 1, len(all_variants)))
            
            for idx, (variant_name, scores) in enumerate(all_variants.items()):
                values = scores + scores[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, label=variant_name, color=colors[idx])
                ax.fill(angles, values, alpha=0.25, color=colors[idx])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([m.upper() for m in metrics])
            ax.set_ylim(0, 1)
            ax.set_title("Performance Radar Chart", size=16, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(output_path / "radar_chart.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_heatmap_comparison(self, results: Dict[str, Any], output_path: Path):
        # Create comprehensive comparison heatmap
        metrics = ["diversity", "fidelity", "privacy", "utility", "overall_score"]
        metric_labels = {
            "diversity": "overall_diversity_score",
            "fidelity": "overall_fidelity_score",
            "privacy": "overall_privacy_score",
            "utility": "overall_utility_score",
            "overall_score": "overall_score"
        }
        
        heatmap_data = []
        variant_labels = []
        
        # FIX: Add baseline (original noise) to heatmap
        if "baselines" in results:
            for variant, data in results["baselines"].items():
                if isinstance(data, dict):
                    row = []
                    for metric in metrics:
                        if metric in data:
                            if metric == "overall_score":
                                row.append(data[metric])
                            else:
                                row.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            row.append(0.0)
                    heatmap_data.append(row)
                    baseline_display_name = variant.replace("_", "+").title()
                    variant_labels.append(f"{baseline_display_name} (B)")
        
        if "conditional" in results:
            for variant, data in results["conditional"].items():
                if isinstance(data, dict):
                    row = []
                    for metric in metrics:
                        if metric in data:
                            if metric == "overall_score":
                                row.append(data[metric])
                            else:
                                row.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            row.append(0.0)
                    heatmap_data.append(row)
                    variant_labels.append(f"{variant} (C)")
        
        if "unconditional" in results:
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict):
                    row = []
                    for metric in metrics:
                        if metric in data:
                            if metric == "overall_score":
                                row.append(data[metric])
                            else:
                                row.append(data[metric].get(metric_labels[metric], 0.0))
                        else:
                            row.append(0.0)
                    heatmap_data.append(row)
                    variant_labels.append(f"{variant} (U)")
        
        if heatmap_data:
            df = pd.DataFrame(heatmap_data, 
                            index=variant_labels,
                            columns=[m.upper().replace("_", " ") for m in metrics])
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, cmap='RdYlGn', center=0.5, 
                       fmt='.3f', cbar_kws={'label': 'Score'})
            plt.title("Comprehensive Performance Heatmap", size=16, fontweight='bold')
            plt.ylabel("Synthetic Variants", size=12)
            plt.xlabel("Evaluation Metrics", size=12)
            plt.tight_layout()
            plt.savefig(output_path / "performance_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_time_series_comparison_plots(self, results: Dict[str, Any], output_path: Path):
        """Create time series comparison plots similar to the take-home assignment"""
        from data_loader import TimeSeriesDataLoader
        
        data_loader = TimeSeriesDataLoader("TimeSeries")
        
        # Create conditional generation comparison plot (4 variants: original, tsv1, tsv2, original_noise)
        self._create_conditional_comparison_plot(data_loader, output_path)
        
        # Create unconditional generation comparison plot (2 variants: original, tsv2)
        self._create_unconditional_comparison_plot(data_loader, output_path)
    
    def _create_conditional_comparison_plot(self, data_loader, output_path: Path):
        """Create comparison plot for conditional generation data"""
        try:
            # Load data for all conditional variants
            _, original_data = data_loader.load_conditional_data("original")
            _, tsv1_data = data_loader.load_conditional_data("tsv1")
            _, tsv2_data = data_loader.load_conditional_data("tsv2")
            _, noise_data = data_loader.load_conditional_data("original_noise")
            
            # Get the first time series (series_id = '0') from each variant
            series_id = '0'
            
            if (series_id in original_data and series_id in tsv1_data and 
                series_id in tsv2_data and series_id in noise_data):
                
                original_ts = original_data[series_id]
                tsv1_ts = tsv1_data[series_id]
                tsv2_ts = tsv2_data[series_id]
                noise_ts = noise_data[series_id]
                
                # Create subplot for each column (excluding OpenInt which is all zeros)
                plot_columns = [col for col in original_ts.columns if col.lower() != 'openint']
                n_cols = len(plot_columns)
                fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
                if n_cols == 1:
                    axes = [axes]
                
                colors = {
                    'Original': '#1f77b4',      # Blue
                    'TSV1': '#ff7f0e',          # Orange  
                    'TSV2': '#2ca02c',          # Green
                    'Original+Noise': '#d62728' # Red
                }
                
                for i, col in enumerate(plot_columns):
                    ax = axes[i]
                    
                    # Plot all variants - show full length where available
                    ax.plot(original_ts[col].values, label='Original', color=colors['Original'], linewidth=2)
                    ax.plot(tsv1_ts[col].values, label='TSV1', color=colors['TSV1'], linewidth=1.5, alpha=0.8)
                    ax.plot(tsv2_ts[col].values, label='TSV2', color=colors['TSV2'], linewidth=1.5, alpha=0.8)
                    ax.plot(noise_ts[col].values, label='Original+Noise', color=colors['Original+Noise'], linewidth=1.5, alpha=0.8)
                    
                    # Add visual indicator for the overlapping region used in metrics
                    min_len = min(len(original_ts), len(tsv1_ts), len(tsv2_ts), len(noise_ts))
                    ax.axvspan(0, min_len-1, alpha=0.1, color='green', label='Evaluation Region')
                    ax.axvline(x=min_len-1, color='red', linestyle='--', alpha=0.7, label=f'Overlap End (t={min_len-1})')
                    
                    ax.set_title(f'Conditional Generation Comparison - {col}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Time Steps')
                    ax.set_ylabel('Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.suptitle(f'Conditional Time Series Comparison (Series {series_id})', fontsize=16, fontweight='bold')
                plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
                plt.savefig(output_path / "conditional_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print("✓ Generated conditional comparison plot")
            else:
                print("⚠ Could not find matching series for conditional comparison")
                
        except Exception as e:
            print(f"Error creating conditional comparison plot: {str(e)}")
    
    def _create_unconditional_comparison_plot(self, data_loader, output_path: Path):
        """Create comparison plot for unconditional generation data matching conditional format"""
        try:
            # Load unconditional data for all available variants (following same format as conditional)
            original_data = data_loader.load_unconditional_data("original")
            tsv2_data = data_loader.load_unconditional_data("tsv2")
            
            # Try to load original_noise for unconditional as well
            try:
                noise_data = data_loader.load_unconditional_data("original_noise")
            except:
                noise_data = None
            
            # Get corresponding time series for fair comparison
            # For unconditional, use the same index/key when possible
            common_keys = set(original_data.keys()) & set(tsv2_data.keys())
            if common_keys:
                # Use a common key if available
                selected_key = list(common_keys)[0]
                original_ts = original_data[selected_key] 
                tsv2_ts = tsv2_data[selected_key]
            else:
                # Fall back to first available from each
                original_key = list(original_data.keys())[0]
                tsv2_key = list(tsv2_data.keys())[0]
                original_ts = original_data[original_key]
                tsv2_ts = tsv2_data[tsv2_key]
            
            # Create subplot for each column (excluding OpenInt which is all zeros)
            plot_columns = [col for col in original_ts.columns if col.lower() != 'openint']
            n_cols = len(plot_columns)
            fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
            if n_cols == 1:
                axes = [axes]
            
            colors = {
                'Original': '#1f77b4',      # Blue
                'TSV2': '#2ca02c',          # Green
                'Original+Noise': '#d62728' # Red
            }
            
            for i, col in enumerate(plot_columns):
                ax = axes[i]
                
                # Plot available variants - show full length for visual comparison
                ax.plot(original_ts[col].values, label='Original', color=colors['Original'], linewidth=2)
                ax.plot(tsv2_ts[col].values, label='TSV2', color=colors['TSV2'], linewidth=1.5, alpha=0.8)
                
                # Add original_noise if available
                if noise_data is not None:
                    if common_keys and selected_key in noise_data:
                        noise_ts = noise_data[selected_key]
                    else:
                        noise_key = list(noise_data.keys())[0]
                        noise_ts = noise_data[noise_key]
                    ax.plot(noise_ts[col].values, label='Original+Noise', color=colors['Original+Noise'], linewidth=1.5, alpha=0.8)
                
                # Add visual indicator for the overlapping region used in metrics
                min_len = min(len(original_ts), len(tsv2_ts))
                if noise_data is not None:
                    min_len = min(min_len, len(noise_ts))
                
                # Shade the overlapping region used for metric calculation
                ax.axvspan(0, min_len-1, alpha=0.1, color='green', label='Evaluation Region')
                ax.axvline(x=min_len-1, color='red', linestyle='--', alpha=0.7, label=f'Overlap End (t={min_len-1})')
                
                ax.set_title(f'Unconditional Generation Comparison - {col}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Time Steps')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Update title to match conditional format
            available_variants = ['Original', 'TSV2']
            if noise_data is not None:
                available_variants.append('Original+Noise')
            plt.suptitle(f'Unconditional Time Series Comparison ({len(available_variants)} variants)', fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
            plt.savefig(output_path / "unconditional_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Generated unconditional comparison plot")
            
        except Exception as e:
            print(f"Error creating unconditional comparison plot: {str(e)}")
    
    def _generate_html_report(self, results: Dict[str, Any], output_path: Path) -> Path:
        html_file = output_path / "evaluation_report.html"
        
        # Extract summary data
        summary = results.get("summary", {})
        best_performers = summary.get("best_performers", {})
        metric_averages = summary.get("metric_averages", {})
        recommendations = summary.get("recommendations", [])
        baseline_performance = summary.get("baseline_performance", {})
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🔍 Time Series Synthetic Data Evaluation Report - Business Analysis</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 10px; }}
                h2 {{ color: #34495e; border-left: 4px solid #3498db; padding-left: 15px; }}
                h3 {{ color: #7f8c8d; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ background-color: #ecf0f1; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #2c3e50; }}
                .metric-label {{ color: #7f8c8d; font-size: 0.9em; text-transform: uppercase; }}
                .recommendation {{ background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .recommendation.excellent {{ background-color: #d4edda; border-color: #c3e6cb; }}
                .recommendation.warning {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .image-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
                .image-container {{ text-align: center; }}
                .image-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                .timestamp {{ text-align: right; color: #7f8c8d; font-size: 0.9em; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #3498db; color: white; }}
                .score-good {{ color: #27ae60; font-weight: bold; }}
                .score-medium {{ color: #f39c12; font-weight: bold; }}
                .score-poor {{ color: #e74c3c; font-weight: bold; }}
                .executive-summary {{ background-color: #e8f4f8; border-left: 5px solid #3498db; padding: 20px; margin: 20px 0; }}
                .critical-finding {{ background-color: #f8d7da; border: 2px solid #dc3545; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .key-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .key-metric {{ background-color: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .immediate-action {{ background-color: #fff3cd; border: 2px solid #ffc107; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .business-impact {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
                .business-card {{ background-color: white; padding: 20px; border-radius: 8px; border-left: 5px solid #dc3545; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .business-card.yellow {{ border-left-color: #ffc107; }}
                .business-card.red {{ border-left-color: #dc3545; }}
                .risk-indicator {{ display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; color: white; }}
                .risk-high {{ background-color: #dc3545; }}
                .risk-medium {{ background-color: #ffc107; color: #000; }}
                .risk-baseline {{ background-color: #28a745; }}
                .issues-section {{ background-color: #fff3cd; border: 1px solid #ffeeba; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                .severe-issues {{ background-color: #f8d7da; border-color: #f5c6cb; }}
                .moderate-issues {{ background-color: #fff3cd; border-color: #ffeeba; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Time Series Synthetic Data Evaluation Report - Business Analysis</h1>
                
                <div class="timestamp">
                    Generated: {
                        results.get('evaluation_metadata', {}).get('timestamp', 'N/A')[:10]
                    }
                    <br> 
                    Author: Jed Lee 
                </div>
                
                <div class="executive-summary">
                    <h2>🎯 Executive Summary</h2>
                    <div class="critical-finding">
                        <strong>CRITICAL FINDING:</strong> Current synthetic data generation methods are <strong style="color: #dc3545;">NOT READY FOR PRODUCTION USE</strong>.
                    </div>
                    
                    <div class="key-metrics">
        """
        
        # Add best performer metrics
        if best_performers:
            for data_type, best in best_performers.items():
                risk_class = "risk-high" if best['score'] < 0.6 else "risk-medium" if best['score'] < 0.8 else "risk-baseline"
                html_content += f"""
                        <div class="key-metric">
                            <strong>Best {data_type.title()} Method ({best['variant'].upper()}):</strong> 
                            <span class="risk-indicator {risk_class}">{best['score']:.1%} HIGH RISK</span>
                        </div>"""
        
        # Add baseline performance
        if baseline_performance:
            for variant, info in baseline_performance.items():
                html_content += f"""
                        <div class="key-metric">
                            <strong>{info['description']}:</strong> 
                            <span class="risk-indicator risk-baseline">{info['score']:.1%} BASELINE</span>
                        </div>"""
        
        html_content += """
                    </div>
                    
                    <div class="immediate-action">
                        <strong>🚨 IMMEDIATE ACTION REQUIRED:</strong> Do not deploy current synthetic data methods in production. 
                        All variants significantly underperform even basic noise addition techniques.
                    </div>
                </div>
                
                <div class="business-impact">
                    <h2>💼 Business Impact Translation</h2>
                    <p><strong>What These Scores Mean for Your Business:</strong></p>
        """
        
        # Use best performer metrics for business impact (prioritize conditional if available)
        diversity_avg = fidelity_avg = privacy_avg = utility_avg = 0
        
        if best_performers:
            # Use conditional best performer if available, otherwise unconditional
            if "conditional" in best_performers:
                best_variant = best_performers["conditional"]["variant"]
                # Find the best performer's detailed results
                if "conditional" in results and best_variant in results["conditional"]:
                    best_results = results["conditional"][best_variant]
                    diversity_avg = best_results.get("diversity", {}).get("overall_diversity_score", 0)
                    fidelity_avg = best_results.get("fidelity", {}).get("overall_fidelity_score", 0)
                    privacy_avg = best_results.get("privacy", {}).get("overall_privacy_score", 0)
                    utility_avg = best_results.get("utility", {}).get("overall_utility_score", 0)
            elif "unconditional" in best_performers:
                best_variant = best_performers["unconditional"]["variant"]
                if "unconditional" in results and best_variant in results["unconditional"]:
                    best_results = results["unconditional"][best_variant]
                    diversity_avg = best_results.get("diversity", {}).get("overall_diversity_score", 0)
                    fidelity_avg = best_results.get("fidelity", {}).get("overall_fidelity_score", 0)
                    privacy_avg = best_results.get("privacy", {}).get("overall_privacy_score", 0)
                    utility_avg = best_results.get("utility", {}).get("overall_utility_score", 0)
        
        # Business impact cards
        html_content += f"""
                    <div class="business-card red">
                        <strong>🔴 Diversity ({diversity_avg:.0%})</strong><br>
                        <strong>Risk:</strong> Synthetic data doesn't cover full business scenarios<br>
                        <strong>Impact:</strong> Models trained on this data will fail on edge cases and new market conditions
                    </div>
                    
                    <div class="business-card yellow">
                        <strong>🟡 Fidelity ({fidelity_avg:.0%})</strong><br>
                        <strong>Risk:</strong> Statistical patterns don't match real data<br>
                        <strong>Impact:</strong> Business analytics and forecasts will be inaccurate
                    </div>
                    
                    <div class="business-card red">
                        <strong>🔴 Privacy ({privacy_avg:.0%})</strong><br>
                        <strong>Risk:</strong> Data may be reverse-engineered to original<br>
                        <strong>Impact:</strong> Potential GDPR/compliance violations, customer trust issues
                    </div>
                    
                    <div class="business-card yellow">
                        <strong>🟡 Utility ({utility_avg:.0%})</strong><br>
                        <strong>Risk:</strong> Data isn't useful for machine learning<br>
                        <strong>Impact:</strong> ML models will perform poorly, wasted development costs
                    </div>
                </div>
                
                <h2>📖 Understanding This Report</h2>
                <p>This evaluation compares synthetic time series data against original data across four key dimensions. 
                <strong>Conditional generation</strong> creates time series based on static features (like categories), while 
                <strong>unconditional generation</strong> creates time series without any conditioning information.</p>
                
                <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
                    <h4>📊 Evaluation Metrics Explained:</h4>
                    <ul>
                        <li><strong>Diversity (25% weight):</strong> Measures how well synthetic data covers the full variety of patterns found in original data. This includes statistical diversity (variance, range, entropy), coverage ratio (what percentage of original data patterns appear in synthetic data), uniqueness score (anti-duplication), and temporal pattern diversity (different time-based trends and autocorrelations). Higher diversity means synthetic data represents the full spectrum of original data characteristics.</li>
                        
                        <li><strong>Fidelity (35% weight):</strong> Assesses how closely synthetic data matches the statistical properties and distributions of original data using core TSGBench metrics. This includes marginal distribution difference (MDD), autocorrelation difference (ACD), statistical moments matching (skewness/kurtosis), dynamic time warping (DTW), and Euclidean distance (ED). Higher fidelity means synthetic data is statistically indistinguishable from original data.</li>
                        
                        <li><strong>Privacy (15% weight):</strong> Evaluates basic privacy risks and memorization detection in synthetic data. This includes distance to closest records (memorization detection) and membership inference vulnerability (basic distinguishability assessment). Privacy is not a core TSGBench focus, so simplified distance-based approaches are used. Higher privacy scores mean better protection against privacy attacks.</li>
                        
                        <li><strong>Utility (25% weight):</strong> Tests practical usefulness and functional equivalence for real-world applications. This includes discriminative scoring (how hard is it to distinguish synthetic from real), predictive performance (forecasting accuracy), downstream task performance (classification/regression), and statistical consistency (business metrics preservation). Higher utility means synthetic data works effectively for practical business purposes.</li>
                    </ul>
                    <p><strong>Baseline:</strong> original_noise is simply the original data with added noise - synthetic methods should ideally outperform this simple baseline to demonstrate meaningful generation capabilities.</p>
                </div>
                
                <h2>🏆 Best Synthetic Performers</h2>
        """
        
        if best_performers:
            html_content += "<ul>"
            for data_type, best in best_performers.items():
                html_content += f"<li><strong>{data_type.upper()}:</strong> {best['variant']} (Score: {best['score']:.4f})</li>"
            html_content += "</ul>"
        
        html_content += """
                <h2>📏 Baseline Performance</h2>
        """
        
        baseline_performance = summary.get("baseline_performance", {})
        if baseline_performance:
            html_content += "<ul>"
            for variant, info in baseline_performance.items():
                html_content += f"<li><strong>{variant.upper()}:</strong> {info['score']:.4f} ({info['description']})</li>"
            html_content += "</ul>"
        
        # Add Critical Technical Issues section 
        html_content += """
                <div class="issues-section severe-issues">
                    <h2>⚠️ Critical Technical Issues Identified</h2>
                    
                    <h3>🔥 Severe Issues (Require Immediate Attention)</h3>
                    <ul>
                        <li><strong>Baseline Underperformance:</strong> All synthetic methods score 19-37% lower than simple noise addition</li>
                        <li><strong>Unconditional Diversity Collapse:</strong> Only 22.8% diversity indicates severe mode collapse</li>
                        <li><strong>Privacy Vulnerabilities:</strong> High membership inference accuracy (95-100%) suggests data memorization</li>
                        <li><strong>Statistical Divergence:</strong> Poor fidelity scores indicate synthetic data distributions don't match original</li>
                    </ul>
                    
                    <h3>🟡 Moderate Issues (Need Investigation)</h3>
                    <ul>
                        <li><strong>Column-Level Inconsistency:</strong> Performance varies significantly across data features</li>
                        <li><strong>Temporal Pattern Loss:</strong> Autocorrelation differences suggest poor time series modeling</li>
                    </ul>
                    
                    <h3>📊 Data Quality Concerns</h3>
                    <ul>
                        <li><strong>Poor Coverage:</strong> Synthetic data covers limited portions of the original data space effectively</li>
                        <li><strong>High Discriminative Accuracy:</strong> ML models can easily distinguish real from synthetic (poor utility)</li>
                        <li><strong>Moment Mismatch:</strong> Skewness and kurtosis differ significantly from original data</li>
                    </ul>
                </div>
        """
        
        html_content += """
                <h2>📊 Performance Summary</h2>
        """
        
        # Add separate summaries for conditional variants in specific order: noise, tsv1, tsv2
        if "conditional" in results or "baselines" in results:
            # Define the desired order: noise first, then tsv1, tsv2
            variant_order = []
            
            # Add noise baseline first
            if "baselines" in results:
                for variant, variant_data in results["baselines"].items():
                    if "noise" in variant.lower() and isinstance(variant_data, dict):
                        variant_order.append(("baseline", variant, variant_data))
            
            # Then add conditional variants in order: tsv1, tsv2
            if "conditional" in results:
                for preferred_variant in ["tsv1", "tsv2"]:
                    for variant, variant_data in results["conditional"].items():
                        if variant.lower() == preferred_variant and isinstance(variant_data, dict):
                            variant_order.append(("conditional", variant, variant_data))
                
                # Add any remaining conditional variants
                for variant, variant_data in results["conditional"].items():
                    if isinstance(variant_data, dict) and variant.lower() not in ["tsv1", "tsv2"]:
                        variant_order.append(("conditional", variant, variant_data))
            
            # Generate summary for each variant in order
            for variant_type, variant, variant_data in variant_order:
                variant_label = f"{variant.upper()} (Baseline)" if variant_type == "baseline" else variant.upper()
                
                # FIX: Add color coding helper function
                def get_score_class(score):
                    if score >= 0.7: return "score-good"
                    elif score >= 0.5: return "score-medium"
                    else: return "score-poor"
                
                # Extract scores with color coding
                overall_score = variant_data.get('overall_score', 0.0)
                diversity_score = variant_data.get('diversity', {}).get('overall_diversity_score', 0.0)
                fidelity_score = variant_data.get('fidelity', {}).get('overall_fidelity_score', 0.0)
                privacy_score = variant_data.get('privacy', {}).get('overall_privacy_score', 0.0)
                utility_score = variant_data.get('utility', {}).get('overall_utility_score', 0.0)
                
                html_content += f"""
                <h3>🔗 Conditional Generation - {variant_label}</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value {get_score_class(overall_score)}">{overall_score:.3f}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {get_score_class(diversity_score)}">{diversity_score:.3f}</div>
                        <div class="metric-label">Diversity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {get_score_class(fidelity_score)}">{fidelity_score:.3f}</div>
                        <div class="metric-label">Fidelity</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {get_score_class(privacy_score)}">{privacy_score:.3f}</div>
                        <div class="metric-label">Privacy</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value {get_score_class(utility_score)}">{utility_score:.3f}</div>
                        <div class="metric-label">Utility</div>
                    </div>
                </div>
                """
        
        if "unconditional" in results:
            for variant, variant_data in results["unconditional"].items():
                if isinstance(variant_data, dict):
                    # FIX: Add color coding helper function (reuse from conditional)
                    def get_score_class(score):
                        if score >= 0.7: return "score-good"
                        elif score >= 0.5: return "score-medium"
                        else: return "score-poor"
                    
                    # Extract scores with color coding
                    overall_score = variant_data.get('overall_score', 0.0)
                    diversity_score = variant_data.get('diversity', {}).get('overall_diversity_score', 0.0)
                    fidelity_score = variant_data.get('fidelity', {}).get('overall_fidelity_score', 0.0)
                    privacy_score = variant_data.get('privacy', {}).get('overall_privacy_score', 0.0)
                    utility_score = variant_data.get('utility', {}).get('overall_utility_score', 0.0)
                    
                    html_content += f"""
                    <h3>🔄 Unconditional Generation - {variant.upper()}</h3>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-value {get_score_class(overall_score)}">{overall_score:.3f}</div>
                            <div class="metric-label">Overall Score</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {get_score_class(diversity_score)}">{diversity_score:.3f}</div>
                            <div class="metric-label">Diversity</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {get_score_class(fidelity_score)}">{fidelity_score:.3f}</div>
                            <div class="metric-label">Fidelity</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {get_score_class(privacy_score)}">{privacy_score:.3f}</div>
                            <div class="metric-label">Privacy</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value {get_score_class(utility_score)}">{utility_score:.3f}</div>
                            <div class="metric-label">Utility</div>
                        </div>
                    </div>
                    """
        
        html_content += """
                <div style="border: 2px solid #28a745; background-color: #f8fff8; padding: 20px; margin: 30px 0; border-radius: 8px;">
                    <h2>🎯 Actionable Recommendations</h2>
                    
                    <div class="issues-section severe-issues">
                        <h3>🚨 Immediate Actions</h3>
                        <ul>
                            <li><strong>STOP production deployment plans</strong> - Current synthetic data is not suitable for business use</li>
                            <li><strong>Establish quality gates</strong> - Define minimum acceptable scores (≥80% vs baseline)</li>
                        </ul>
                    </div>
                    
                    <div class="issues-section moderate-issues">
                        <h3>🔧 Technical Improvements</h3>
                        <ul>
                            <li><strong>Algorithm review</strong> - Investigate why methods underperform simple noise addition</li>
                            <li><strong>Address diversity collapse</strong> - Fix unconditional generation variety issues</li>
                            <li><strong>Improve privacy</strong> - Implement differential privacy and reduce memorization</li>
                        </ul>
                    </div>
                    
                    <div class="issues-section">
                        <h3>📋 Next Steps</h3>
                        <ul>
                            <li><strong>Validation framework</strong> - Implement automated quality testing</li>
                            <li><strong>Business requirements</strong> - Define specific use-case needs</li>
                            <li><strong>Performance monitoring</strong> - Set up continuous data quality monitoring</li>
                        </ul>
                    </div>
                </div>
        """
        
        # Detailed recommendations section removed as requested
        
        html_content += """
                <h2>📈 Performance Visualizations</h2>
                
                <div class="image-container" style="margin-bottom: 30px;">
                    <h3>Metric Breakdown</h3>
                    <img src="metric_breakdown.png" alt="Metric Breakdown Chart" style="width: 100%; max-width: 1000px;">
                </div>
                
                <div class="image-container" style="margin-bottom: 30px;">
                    <h3>Performance Radar</h3>
                    <img src="radar_chart.png" alt="Radar Chart" style="width: 100%; max-width: 800px;">
                </div>
                
                <div class="image-container" style="margin-bottom: 30px;">
                    <h3>Comprehensive Heatmap</h3>
                    <img src="performance_heatmap.png" alt="Performance Heatmap" style="width: 100%; max-width: 1000px;">
                </div>
                
                <h2>📊 Time Series Data Comparison & Analysis</h2>
                <p>Visual comparison and detailed performance analysis showing how synthetic methods compare to original data patterns.</p>
                
                
                <div class="image-container" style="margin-bottom: 30px;">
                    <h3>🔗 Conditional Generation - Overall Comparison</h3>
                    <p>Shows all 4 conditional variants (Original, TSV1, TSV2, Original+Noise) overlaid for direct comparison.</p>
                    <img src="conditional_comparison.png" alt="Conditional Generation Time Series Comparison" style="width: 100%; max-width: 1200px;">
                </div>
        """
        
        # Add integrated conditional analysis (plot + tables combined)
        html_content += self._generate_integrated_conditional_analysis(results)
        
        html_content += """
                <div class="image-container" style="margin-bottom: 30px;">
                    <h3>🔄 Unconditional Generation - Overall Comparison</h3>
                    <p>Shows all available unconditional variants overlaid for direct comparison.</p>
                    
                    <div style="background-color: #fff3cd; border: 1px solid #ffeeba; padding: 15px; margin: 15px 0; border-radius: 5px;">
                        <strong>⚠️ Data Length Adjustment:</strong> This evaluation uses only overlapping timesteps for fair comparison. 
                        Metrics are calculated only on the common time period where all variants have actual data, without any padding or truncation.
                        Visualizations may show the full length of each series, but scoring reflects only the overlapping portion.
                    </div>
                    
                    <img src="unconditional_comparison.png" alt="Unconditional Generation Time Series Comparison" style="width: 100%; max-width: 1200px;">
                </div>
        """
        
        # Add integrated unconditional analysis (plot + tables combined)  
        html_content += self._generate_integrated_unconditional_analysis(results)
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        return html_file
    
    def _generate_column_breakdown_tables(self, results: Dict[str, Any]) -> str:
        """Generate detailed column-by-column performance breakdown tables with proper structure."""
        html_content = ""
        
        # Check if we have column-level data 
        has_column_data = False
        if results.get("conditional"):
            first_variant = list(results["conditional"].values())[0]
            if isinstance(first_variant, dict) and "diversity" in first_variant:
                diversity_data = first_variant["diversity"]
                has_column_data = any("_diversity_score" in key for key in diversity_data.keys())
        
        if not has_column_data:
            return ""  # Skip if no column data available
        
        # Column Performance Overview
        html_content += """
        <h3>💎 Per-Column Performance Analysis</h3>
        <p>This section shows how each individual column performs across all evaluation metrics. Each column is evaluated separately 
        for diversity, fidelity, privacy, and utility. This helps identify specific features that need targeted improvements.</p>
        
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 8px; margin: 15px 0;">
            <h5>📊 New Table Structure:</h5>
            <ul>
                <li><strong>Each Row = One Column:</strong> Individual financial features (Col1/Col2/Col3 or Open/High/Low/Close/Volume)</li>
                <li><strong>Each Column = One Metric:</strong> Diversity, Fidelity, Privacy, Utility scores for that specific feature</li>
                <li><strong>Per-Column Analysis:</strong> Shows how well each financial dimension is synthesized independently</li>
            </ul>
        </div>
        """
        
        # Generate conditional column breakdown
        if "conditional" in results:
            html_content += self._generate_conditional_column_table(results["conditional"])
        
        # Generate unconditional column breakdown  
        if "unconditional" in results:
            html_content += self._generate_unconditional_column_table(results["unconditional"])
        
        # Generate baseline column breakdown
        if "baselines" in results:
            html_content += self._generate_baseline_column_table(results["baselines"])
            
        # Add comprehensive interpretation guide
        html_content += """
        <div style="background-color: #f0f8ff; padding: 15px; border-radius: 8px; margin: 20px 0;">
            <h5>📖 How to Read These New Tables:</h5>
            <ul>
                <li><strong>Row Structure:</strong> Each row represents one financial feature (column) being evaluated</li>
                <li><strong>Column Structure:</strong> Each column shows a different evaluation metric for that feature</li>
                <li><strong>Per-Feature Analysis:</strong> Compare scores across rows to see which features perform better/worse</li>
                <li><strong>Per-Metric Analysis:</strong> Compare scores within columns to see metric-specific patterns</li>
                <li><strong>Color Coding:</strong> <span class="score-good">Green ≥ 0.7:</span> Excellent, 
                    <span class="score-medium">Yellow ≥ 0.5:</span> Acceptable, 
                    <span class="score-poor">Red < 0.5:</span> Needs improvement</li>
            </ul>
            <p><strong>Key Insights:</strong> 
            • Consistent colors across a row indicate balanced performance for that feature<br>
            • Consistent colors down a column indicate that metric performs similarly across all features<br>
            • Red cells pinpoint specific feature-metric combinations needing attention</p>
        </div>
        """
        
        return html_content
    
    def _generate_conditional_column_analysis(self, results: Dict[str, Any]) -> str:
        """Generate conditional per-column analysis with column-first structure."""
        html_content = """
                <h2>🔗 Conditional Generation - Per-Column Performance Analysis</h2>
                <p>Column-by-column comparison across all conditional methods (Original+Noise, TSV1, TSV2).</p>
        """
        
        # Get all available columns from conditional data
        available_columns = set()
        all_methods = {}
        
        # Collect baseline data
        if "baselines" in results:
            for variant, data in results["baselines"].items():
                if "noise" in variant.lower() and isinstance(data, dict):
                    baseline_name = variant.replace("_", "+").title()
                    all_methods[f"{baseline_name} (Baseline)"] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Collect conditional data
        if "conditional" in results:
            for variant, data in results["conditional"].items():
                if isinstance(data, dict):
                    all_methods[variant.upper()] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Sort columns for consistent ordering
        available_columns = sorted(list(available_columns))
        
        # Generate column-by-column comparison tables
        for column in available_columns:
            html_content += self._generate_single_column_comparison_table(column, all_methods, "conditional")
        
        return html_content
    
    def _generate_unconditional_column_analysis(self, results: Dict[str, Any]) -> str:
        """Generate unconditional per-column analysis with column-first structure.""" 
        html_content = """
                <h2>🔄 Unconditional Generation - Per-Column Performance Analysis</h2>
                <p>Column-by-column comparison across all unconditional methods.</p>
        """
        
        # Get all available columns from unconditional data
        available_columns = set()
        all_methods = {}
        
        # Collect unconditional data
        if "unconditional" in results:
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict):
                    all_methods[variant.upper()] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Sort columns for consistent ordering (financial columns)
        column_order = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in column_order if col in available_columns]
        
        # Generate column-by-column comparison tables
        for column in available_columns:
            html_content += self._generate_single_column_comparison_table(column, all_methods, "unconditional")
        
        return html_content
    
    def _generate_integrated_conditional_analysis(self, results: Dict[str, Any]) -> str:
        """Generate integrated conditional analysis with per-column tables immediately after the plot."""
        html_content = """
                <h3>📊 Conditional Generation - Detailed Column Analysis</h3>
                <p>Performance breakdown for each column (col1, col2, col3) comparing Original+Noise baseline with synthetic methods.</p>
        """
        
        # Get all available columns and methods  
        available_columns = set()
        all_methods = {}
        
        # Collect baseline data
        if "baselines" in results:
            for variant, data in results["baselines"].items():
                if "noise" in variant.lower() and isinstance(data, dict):
                    baseline_name = variant.replace("_", "+").title()
                    all_methods[f"{baseline_name} (Baseline)"] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Collect conditional data
        if "conditional" in results:
            for variant, data in results["conditional"].items():
                if isinstance(data, dict):
                    all_methods[variant.upper()] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Sort columns for consistent ordering
        available_columns = sorted(list(available_columns))
        
        # Generate column-by-column comparison tables
        for column in available_columns:
            html_content += self._generate_single_column_comparison_table(column, all_methods, "conditional")
        
        return html_content
    
    def _generate_integrated_unconditional_analysis(self, results: Dict[str, Any]) -> str:
        """Generate integrated unconditional analysis with per-column tables immediately after the plot."""
        html_content = """
                <h3>📊 Unconditional Generation - Detailed Column Analysis</h3>
                <p>Performance breakdown for each financial column (Open, High, Low, Close, Volume).</p>
        """
        
        # Get all available columns and methods
        available_columns = set()
        all_methods = {}
        
        # Collect unconditional data
        if "unconditional" in results:
            for variant, data in results["unconditional"].items():
                if isinstance(data, dict):
                    all_methods[variant.upper()] = data
                    if 'diversity' in data:
                        for key in data['diversity'].keys():
                            if key.endswith('_diversity_score') and key != 'overall_diversity_score':
                                available_columns.add(key.replace('_diversity_score', ''))
        
        # Sort columns for consistent ordering (financial columns)
        column_order = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in column_order if col in available_columns]
        
        # Generate column-by-column comparison tables
        for column in available_columns:
            html_content += self._generate_single_column_comparison_table(column, all_methods, "unconditional")
        
        return html_content
    
    def _generate_single_column_comparison_table(self, column: str, all_methods: Dict[str, Dict], data_type: str) -> str:
        """Generate comparison table for a single column across all methods."""
        
        def get_score_class(score):
            if score >= 0.7: return "score-good"
            elif score >= 0.5: return "score-medium"
            else: return "score-poor"
        
        def format_score(score):
            if score < 0.001:
                return f"{score:.2e}"
            else:
                return f"{score:.3f}"
        
        html_content = f"""
        <h4>📊 {column} - Method Comparison</h4>
        <table>
            <tr>
                <th>Method</th>
                <th>Diversity</th>
                <th>Fidelity</th>
                <th>Privacy</th>
                <th>Utility</th>
            </tr>
        """
        
        # Order methods: Baseline first, then alphabetically
        method_order = []
        baseline_methods = [method for method in all_methods.keys() if "Baseline" in method]
        synthetic_methods = [method for method in all_methods.keys() if "Baseline" not in method]
        method_order.extend(sorted(baseline_methods))
        method_order.extend(sorted(synthetic_methods))
        
        for method_name in method_order:
            method_data = all_methods[method_name]
            
            # Extract scores for this column
            diversity_score = self._extract_column_score(method_data, column, "diversity")
            fidelity_score = self._extract_column_score(method_data, column, "fidelity")
            privacy_score = self._extract_column_score(method_data, column, "privacy")
            utility_score = self._extract_column_score(method_data, column, "utility")
            
            html_content += f"""
            <tr>
                <td><strong>{method_name}</strong></td>
                <td class="{get_score_class(diversity_score)}">{format_score(diversity_score)}</td>
                <td class="{get_score_class(fidelity_score)}">{format_score(fidelity_score)}</td>
                <td class="{get_score_class(privacy_score)}">{format_score(privacy_score)}</td>
                <td class="{get_score_class(utility_score)}">{format_score(utility_score)}</td>
            </tr>
            """
        
        html_content += "</table><br><br>"
        return html_content
    
    def _generate_conditional_column_table(self, conditional_results: Dict[str, Any]) -> str:
        """Generate column-wise table for conditional generation results."""
        html_content = """
        <h4>🔗 Conditional Generation - Per-Column Performance</h4>
        """
        
        # Get available columns from the first variant
        first_variant = list(conditional_results.values())[0]
        available_columns = self._extract_available_columns(first_variant, "conditional")
        
        if not available_columns:
            return html_content + "<p>No column-wise data available for conditional generation.</p>"
        
        for variant_name, variant_results in conditional_results.items():
            if not isinstance(variant_results, dict):
                continue
                
            html_content += f"""
            <h5>{variant_name.upper()} - Individual Column Performance</h5>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Diversity</th>
                    <th>Fidelity</th>
                    <th>Privacy</th>
                    <th>Utility</th>
                </tr>
            """
            
            for column in available_columns:
                # Extract metrics for this column
                diversity_score = self._extract_column_score(variant_results, column, "diversity")
                fidelity_score = self._extract_column_score(variant_results, column, "fidelity")  
                privacy_score = self._extract_column_score(variant_results, column, "privacy")
                utility_score = self._extract_column_score(variant_results, column, "utility")
                
                def score_class(score):
                    if score >= 0.7: return "score-good"
                    elif score >= 0.5: return "score-medium" 
                    else: return "score-poor"
                
                # Better formatting for very small numbers
                def format_score(score):
                    if score < 0.001:
                        return f"{score:.2e}"  # Scientific notation for very small numbers
                    else:
                        return f"{score:.3f}"  # Normal formatting
                
                html_content += f"""
                <tr>
                    <td><strong>{column}</strong></td>
                    <td class="{score_class(diversity_score)}">{format_score(diversity_score)}</td>
                    <td class="{score_class(fidelity_score)}">{format_score(fidelity_score)}</td>
                    <td class="{score_class(privacy_score)}">{format_score(privacy_score)}</td>
                    <td class="{score_class(utility_score)}">{format_score(utility_score)}</td>
                </tr>
                """
            
            # FIX: Add overall scores at the bottom of each table
            overall_diversity = variant_results.get('diversity', {}).get('overall_diversity_score', 0.0)
            overall_fidelity = variant_results.get('fidelity', {}).get('overall_fidelity_score', 0.0)
            overall_privacy = variant_results.get('privacy', {}).get('overall_privacy_score', 0.0)
            overall_utility = variant_results.get('utility', {}).get('overall_utility_score', 0.0)
            
            html_content += f"""
                <tr style="border-top: 2px solid #333; background-color: #f8f9fa;">
                    <td><strong>OVERALL</strong></td>
                    <td class="{score_class(overall_diversity)}" style="font-weight: bold;">{format_score(overall_diversity)}</td>
                    <td class="{score_class(overall_fidelity)}" style="font-weight: bold;">{format_score(overall_fidelity)}</td>
                    <td class="{score_class(overall_privacy)}" style="font-weight: bold;">{format_score(overall_privacy)}</td>
                    <td class="{score_class(overall_utility)}" style="font-weight: bold;">{format_score(overall_utility)}</td>
                </tr>
            </table><br>"""
            
        return html_content
    
    def _generate_unconditional_column_table(self, unconditional_results: Dict[str, Any]) -> str:
        """Generate column-wise table for unconditional generation results."""
        html_content = """
        <h4>🔄 Unconditional Generation - Per-Column Performance</h4>
        """
        
        # Get available columns from the first variant
        first_variant = list(unconditional_results.values())[0]
        available_columns = self._extract_available_columns(first_variant, "unconditional")
        
        if not available_columns:
            return html_content + "<p>No column-wise data available for unconditional generation.</p>"
        
        for variant_name, variant_results in unconditional_results.items():
            if not isinstance(variant_results, dict):
                continue
                
            html_content += f"""
            <h5>{variant_name.upper()} - Individual Column Performance</h5>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Diversity</th>
                    <th>Fidelity</th>
                    <th>Privacy</th>
                    <th>Utility</th>
                </tr>
            """
            
            for column in available_columns:
                # Extract metrics for this column
                diversity_score = self._extract_column_score(variant_results, column, "diversity")
                fidelity_score = self._extract_column_score(variant_results, column, "fidelity")  
                privacy_score = self._extract_column_score(variant_results, column, "privacy")
                utility_score = self._extract_column_score(variant_results, column, "utility")
                
                def score_class(score):
                    if score >= 0.7: return "score-good"
                    elif score >= 0.5: return "score-medium" 
                    else: return "score-poor"
                
                # Better formatting for very small numbers
                def format_score(score):
                    if score < 0.001:
                        return f"{score:.2e}"  # Scientific notation for very small numbers
                    else:
                        return f"{score:.3f}"  # Normal formatting
                
                html_content += f"""
                <tr>
                    <td><strong>{column}</strong></td>
                    <td class="{score_class(diversity_score)}">{format_score(diversity_score)}</td>
                    <td class="{score_class(fidelity_score)}">{format_score(fidelity_score)}</td>
                    <td class="{score_class(privacy_score)}">{format_score(privacy_score)}</td>
                    <td class="{score_class(utility_score)}">{format_score(utility_score)}</td>
                </tr>
                """
                
            # FIX: Add overall scores at the bottom of each table
            overall_diversity = variant_results.get('diversity', {}).get('overall_diversity_score', 0.0)
            overall_fidelity = variant_results.get('fidelity', {}).get('overall_fidelity_score', 0.0)
            overall_privacy = variant_results.get('privacy', {}).get('overall_privacy_score', 0.0)
            overall_utility = variant_results.get('utility', {}).get('overall_utility_score', 0.0)
            
            html_content += f"""
                <tr style="border-top: 2px solid #333; background-color: #f8f9fa;">
                    <td><strong>OVERALL</strong></td>
                    <td class="{score_class(overall_diversity)}" style="font-weight: bold;">{format_score(overall_diversity)}</td>
                    <td class="{score_class(overall_fidelity)}" style="font-weight: bold;">{format_score(overall_fidelity)}</td>
                    <td class="{score_class(overall_privacy)}" style="font-weight: bold;">{format_score(overall_privacy)}</td>
                    <td class="{score_class(overall_utility)}" style="font-weight: bold;">{format_score(overall_utility)}</td>
                </tr>
            </table><br>"""
            
        return html_content
        
    def _generate_baseline_column_table(self, baseline_results: Dict[str, Any]) -> str:
        """Generate column-wise table for baseline results.""" 
        html_content = """
        <h4>📏 Baseline - Per-Column Performance</h4>
        """
        
        # Get available columns from the first baseline
        first_baseline = list(baseline_results.values())[0]
        available_columns = self._extract_available_columns(first_baseline, "baseline")
        
        if not available_columns:
            return html_content + "<p>No column-wise data available for baseline.</p>"
        
        for baseline_name, baseline_results_data in baseline_results.items():
            if not isinstance(baseline_results_data, dict):
                continue
                
            html_content += f"""
            <h5>{baseline_name.upper()} - Individual Column Performance</h5>
            <table>
                <tr>
                    <th>Column</th>
                    <th>Diversity</th>
                    <th>Fidelity</th>
                    <th>Privacy</th>
                    <th>Utility</th>
                </tr>
            """
            
            for column in available_columns:
                # Extract metrics for this column
                diversity_score = self._extract_column_score(baseline_results_data, column, "diversity")
                fidelity_score = self._extract_column_score(baseline_results_data, column, "fidelity")  
                privacy_score = self._extract_column_score(baseline_results_data, column, "privacy")
                utility_score = self._extract_column_score(baseline_results_data, column, "utility")
                
                def score_class(score):
                    if score >= 0.7: return "score-good"
                    elif score >= 0.5: return "score-medium" 
                    else: return "score-poor"
                
                # Better formatting for very small numbers
                def format_score(score):
                    if score < 0.001:
                        return f"{score:.2e}"  # Scientific notation for very small numbers
                    else:
                        return f"{score:.3f}"  # Normal formatting
                
                html_content += f"""
                <tr>
                    <td><strong>{column}</strong></td>
                    <td class="{score_class(diversity_score)}">{format_score(diversity_score)}</td>
                    <td class="{score_class(fidelity_score)}">{format_score(fidelity_score)}</td>
                    <td class="{score_class(privacy_score)}">{format_score(privacy_score)}</td>
                    <td class="{score_class(utility_score)}">{format_score(utility_score)}</td>
                </tr>
                """
                
            # FIX: Add overall scores at the bottom of each table
            overall_diversity = baseline_results_data.get('diversity', {}).get('overall_diversity_score', 0.0)
            overall_fidelity = baseline_results_data.get('fidelity', {}).get('overall_fidelity_score', 0.0)
            overall_privacy = baseline_results_data.get('privacy', {}).get('overall_privacy_score', 0.0)
            overall_utility = baseline_results_data.get('utility', {}).get('overall_utility_score', 0.0)
            
            html_content += f"""
                <tr style="border-top: 2px solid #333; background-color: #f8f9fa;">
                    <td><strong>OVERALL</strong></td>
                    <td class="{score_class(overall_diversity)}" style="font-weight: bold;">{format_score(overall_diversity)}</td>
                    <td class="{score_class(overall_fidelity)}" style="font-weight: bold;">{format_score(overall_fidelity)}</td>
                    <td class="{score_class(overall_privacy)}" style="font-weight: bold;">{format_score(overall_privacy)}</td>
                    <td class="{score_class(overall_utility)}" style="font-weight: bold;">{format_score(overall_utility)}</td>
                </tr>
            </table><br>"""
            
        return html_content
    
    def _extract_available_columns(self, variant_results: Dict[str, Any], data_type: str) -> List[str]:
        """Extract available column names from variant results."""
        columns = []
        
        # Check diversity data for column names
        diversity_data = variant_results.get('diversity', {})
        if diversity_data:
            # Look for column-specific diversity scores
            for key in diversity_data.keys():
                if key.endswith('_diversity_score'):
                    col_name = key.replace('_diversity_score', '')
                    if col_name not in ['overall']:  # Skip overall metrics
                        columns.append(col_name)
        
        # Ensure consistent ordering
        if any(col in columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
            # Financial columns for unconditional data
            financial_order = ['Open', 'High', 'Low', 'Close', 'Volume']
            columns = [col for col in financial_order if col in columns]
        elif any(col in columns for col in ['col1', 'col2', 'col3']):
            # Generic columns for conditional data  
            generic_order = ['col1', 'col2', 'col3']
            columns = [col for col in generic_order if col in columns]
        
        return columns
    
    def _extract_column_score(self, variant_results: Dict[str, Any], column: str, metric_type: str) -> float:
        """Extract score for specific column and metric type."""
        if metric_type == "diversity":
            return variant_results.get('diversity', {}).get(f'{column}_diversity_score', 0.0)
        elif metric_type == "fidelity":
            column_fidelity = variant_results.get('column_fidelity', {}).get(column, {})
            return column_fidelity.get('column_fidelity_score', 0.0)
        elif metric_type == "privacy":
            column_privacy = variant_results.get('column_privacy', {}).get(column, {})
            return column_privacy.get('overall_privacy_score', 0.0)
        elif metric_type == "utility":
            column_utility = variant_results.get('column_utility', {}).get(column, {})
            return column_utility.get('overall_utility_score', 0.0)
        else:
            return 0.0
    
    def _generate_business_summary(self, results: Dict[str, Any], output_path: Path):
        summary_file = output_path / "business_summary.txt"
        
        summary = results.get("summary", {})
        metric_averages = summary.get("metric_averages", {})
        best_performers = summary.get("best_performers", {})
        baseline_performance = summary.get("baseline_performance", {})
        recommendations = summary.get("recommendations", {})
        
        with open(summary_file, 'w') as f:
            f.write("SYNTHETIC TIME SERIES DATA EVALUATION - BUSINESS SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            
            # Calculate overall performance across both types
            overall_conditional = metric_averages.get("conditional", {}).get("overall", 0.0)
            overall_unconditional = metric_averages.get("unconditional", {}).get("overall", 0.0)
            overall_avg = (overall_conditional + overall_unconditional) / 2 if overall_conditional > 0 and overall_unconditional > 0 else max(overall_conditional, overall_unconditional)
            
            f.write(f"Overall Synthetic Performance: {overall_avg:.1%}\n")
            f.write(f"Data Quality Rating: {'Excellent' if overall_avg > 0.7 else 'Good' if overall_avg > 0.5 else 'Needs Improvement'}\n\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-" * 20 + "\n")
            for data_type, best in best_performers.items():
                f.write(f"Best {data_type} variant: {best['variant']} ({best['score']:.1%})\n")
            
            # Add baseline performance
            for variant, info in baseline_performance.items():
                f.write(f"Baseline ({variant}): {info['score']:.1%} ({info['description']})\n")
            f.write("\n")
            
            f.write("PERFORMANCE BY GENERATION TYPE\n")
            f.write("-" * 35 + "\n")
            
            if "conditional" in metric_averages:
                f.write("CONDITIONAL GENERATION:\n")
                cond_metrics = metric_averages["conditional"]
                for metric, value in cond_metrics.items():
                    f.write(f"  {metric.capitalize()}: {value:.1%}\n")
                f.write("\n")
            
            if "unconditional" in metric_averages:
                f.write("UNCONDITIONAL GENERATION:\n")
                uncond_metrics = metric_averages["unconditional"]
                for metric, value in uncond_metrics.items():
                    f.write(f"  {metric.capitalize()}: {value:.1%}\n")
                f.write("\n")
            
            f.write("RECOMMENDATIONS FOR BUSINESS USE\n")
            f.write("-" * 35 + "\n")
            
            # Handle new recommendation structure
            if isinstance(recommendations, dict):
                for category, recs in recommendations.items():
                    if recs:
                        f.write(f"{category.upper().replace('_', ' ')}:\n")
                        for rec in recs:
                            f.write(f"• {rec}\n")
                        f.write("\n")
            else:
                # Fallback for old format
                for rec in recommendations:
                    f.write(f"• {rec}\n")

def main():
    # Load the most recent evaluation results
    import glob
    result_files = glob.glob("evaluation_results_*.json")
    if not result_files:
        print("No evaluation results found. Please run evaluation_pipeline.py first.")
        return
    
    latest_file = max(result_files)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    generator = ReportGenerator()
    report_path = generator.generate_comprehensive_report(results)
    
    print(f"Comprehensive report generated: {report_path}")

if __name__ == "__main__":
    main()