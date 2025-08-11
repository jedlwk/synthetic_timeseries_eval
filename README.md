# 🔬 Synthetic Time Series Evaluation Framework

A **TSGBench-compliant evaluation framework** for assessing conditional and unconditional synthetic time series across four key dimensions: **Fidelity**, **Diversity**, **Utility**, and **Privacy**.

## 🎯 Overview

This framework implements the **TSGBench methodology** (VLDB 2024 Best Paper Award Nominee) for standardized, reproducible evaluation of synthetic time series generation methods.

### ✨ Key Features

- **🏆 TSGBench-Aligned**: Implements core TSGBench metrics (MDD, ACD, DTW, DS, PS)
- **📊 Standardized Evaluation**: Research-compliant methodology ensuring reproducible results
- **🤖 Automated Pipeline**: End-to-end evaluation with minimal configuration
- **📈 Professional Reports**: HTML dashboards with strategic visualizations and business translations
- **⚡ Production-Ready**: Handles enterprise-scale multivariate time series with overlap-based comparison

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn dtaidistance
```

### Basic Usage

```python
from evaluation_pipeline import TimeSeriesEvaluationPipeline

# Initialize the pipeline
pipeline = TimeSeriesEvaluationPipeline("path/to/TimeSeries")

# Evaluate all synthetic variants
results = pipeline.evaluate_all_variants()

# Generate comprehensive report
from report_generator import ReportGenerator
generator = ReportGenerator()
report_path = generator.generate_comprehensive_report(results)
```

### Command Line Usage

```bash
# Run complete evaluation pipeline
python3 evaluation_pipeline.py

# Generate visual reports
python3 report_generator.py
```

## 📁 Data Structure Expected

```
synthetic_ts_eval/
├── TimeSeries/
│   ├── conditional generation/
│   │   ├── original/                # Ground truth
│   │   │   ├── static.csv          # Static conditions (.id, features)  
│   │   │   └── time_series/        # Time series files (series_XXX.csv)
│   │   ├── tsv1/                   # Synthetic variant 1
│   │   │   ├── static.csv
│   │   │   └── time_series/        # Time series files (sample_X.csv)
│   │   ├── tsv2/                   # Synthetic variant 2
│   │   │   ├── static.csv
│   │   │   └── time_series/        # Time series files (X.csv)
│   │   └── original_noise/         # Baseline (original + noise)
│   └── unconditional generation/
│       ├── original/               # Ground truth time series (stock data)
│       └── tsv2/                   # Synthetic variant
│           └── ts/                 # Time series files (X.csv)
├── evaluation_report/              # Generated HTML reports and visualizations
│   ├── evaluation_report.html
│   ├── conditional_comparison.png
│   ├── unconditional_comparison.png
│   ├── overall_performance.png
│   ├── metric_breakdown.png
│   ├── performance_heatmap.png
│   └── radar_chart.png
├── data_loader.py                  # Data loading utilities
├── diversity_metrics.py            # Diversity evaluation metrics
├── fidelity_metrics.py            # Fidelity evaluation metrics  
├── privacy_metrics.py             # Privacy evaluation metrics
├── utility_metrics.py             # Utility evaluation metrics
├── evaluation_pipeline.py         # Main evaluation orchestrator
├── report_generator.py            # Report generation system
└── README.md                      # This documentation
```

## 📈 Current TSGBench Evaluation Results

### 🚨 CRITICAL FINDING: NOT READY FOR PRODUCTION
**Overall Synthetic Performance: 47.5% (Conditional) | 23.4% (Unconditional)**  
**Data Quality Rating: HIGH RISK - Major Improvements Required**

### Best Performing Synthetic Variants:
- **Conditional Generation**: TSV2 (57.0%) - **HIGH RISK** (37% below baseline)
- **Unconditional Generation**: TSV2 (23.4%) - **CRITICAL FAILURE** (50% below baseline)

### TSGBench Baseline Performance:
- **Original+Noise**: 72.9% (Simple Gaussian noise significantly outperforms all synthetic methods)

### Performance by Generation Type:

#### 🔗 Conditional Generation (Generic Time Series):
- **Overall Score**: 47.5% - **HIGH RISK** for production deployment
- **Diversity**: 54.7% (moderate - pattern coverage gaps)
- **Fidelity**: 47.4% (poor - TSGBench metrics show statistical mismatch)
- **Privacy**: 34.6% (high risk - 92-100% membership inference accuracy)
- **Utility**: 48.3% (poor - limited downstream task performance)

#### 🔄 Unconditional Generation (Financial OHLC-V Data):
- **Overall Score**: 23.4% - **CRITICAL FAILURE** for any business application
- **Diversity**: 22.8% (critical - severe mode collapse across all financial features) 
- **Fidelity**: 12.1% (critical - fundamental distributional failures)
- **Privacy**: 45.4% (moderate - ironically better due to poor generation quality)
- **Utility**: 26.7% (critical - unsuitable for financial modeling)

### 🚨 TSGBench-Identified Critical Issues:
```
🔥 CRITICAL FAILURES:
• Baseline Underperformance: All synthetic methods score 37-50% lower than simple noise
• Mode Collapse: Unconditional diversity (22.8%) indicates severe pattern limitation  
• Data Memorization: 92-100% membership inference accuracy (privacy failure)
• Statistical Mismatch: Core TSGBench metrics show poor distributional alignment

⚠️ SECONDARY CONCERNS:  
• Financial Constraints Violation: OHLC relationships not preserved in unconditional data
• Temporal Pattern Loss: Autocorrelation and time series structure degraded
• Feature Inconsistency: Column-level performance varies significantly (12-85% range)

📊 TSGBENCH METRIC FAILURES:
• MDD (Marginal Distribution Difference): Poor histogram-based alignment
• ACD (Autocorrelation Difference): Temporal structure degradation  
• DS (Discriminative Score): Easy distinguishability (|0.5 - accuracy| too high)
• PS (Predictive Score): Poor downstream task performance
```

### TSGBench-Compliant Evaluation Reports

- **Comprehensive HTML Report**: `evaluation_report/evaluation_report.html`
  - TSGBench methodology overview and metric explanations
  - Executive summary with production readiness assessment  
  - Detailed conditional/unconditional generation analysis
  - Business impact translation with risk categorization
  - Column-level performance breakdown (financial OHLC-V vs. generic features)
  
- **Professional Visualizations**: 
  - TSGBench metric breakdown with baseline comparisons
  - Radar charts showing four-dimensional assessment
  - Performance heatmaps highlighting critical gaps
  - Time series comparison plots with temporal alignment methodology
  - Column-specific analysis tables with score interpretations

## 🏗️ Architecture

### Core Components

1. **DataLoader** (`data_loader.py`): TSGBench-compliant data loading with conditional/unconditional separation and overlap-based temporal alignment
2. **TSGBench Metric Evaluators**: Exact implementation of core TSGBench methodology
   - `fidelity_metrics.py`: MDD (histogram-based), ACD, statistical moments, DTW, Euclidean distance
   - `utility_metrics.py`: DS (|0.5 - accuracy|), PS (train synthetic, test original), C-FID approximation  
   - `diversity_metrics.py`: Coverage analysis, uniqueness scoring, statistical diversity with domain weighting
   - `privacy_metrics.py`: **Supplementary analysis** - DCR and MIR assessment (not core TSGBench)
3. **Pipeline** (`evaluation_pipeline.py`): Orchestrates TSGBench evaluation with scale normalization and error handling
4. **Reporter** (`report_generator.py`): Generates production-ready reports with business impact analysis and technical deep-dive

### Extensibility

The framework is designed for easy extension:

```python
class CustomMetric:
    def compute_metric(self, original_data, synthetic_data):
        # Implement custom evaluation logic
        return {"custom_score": score}

# Add to pipeline
pipeline.custom_evaluator = CustomMetric()
```

## 🔧 Configuration

### TSGBench-Compliant Metric Weights

Weighting scheme reflecting TSGBench methodology priorities with supplementary privacy analysis:

```python
weights = {
    "fidelity": 0.35,    # Core TSGBench focus: MDD, ACD, DTW, ED, statistical moments
    "diversity": 0.25,   # Extended TSGBench: coverage, uniqueness, statistical variety
    "utility": 0.25,     # TSGBench model-based: DS, PS, downstream task performance
    "privacy": 0.15      # Supplementary (not TSGBench): DCR, MIR assessment
}

# TSGBench Core Metrics Implementation:
tsgbench_core = {
    "feature_based": ["MDD", "ACD", "SD", "KD"],      # Statistical properties
    "distance_based": ["DTW", "ED"],                   # Sequence similarity  
    "model_based": ["DS", "PS", "C-FID"]              # ML task performance
}
```

## 📚 References

1. **Ang, Yihao, et al.** "TSGBench: Time Series Generation Benchmark." *VLDB Endowment* 17.3 (2024): 305-318 *(Primary methodology source)*
2. **Salvador, Stan & Chan, Philip.** "FastDTW: Toward Accurate Dynamic Time Warping." *KDD* (2007)
3. **Shokri, Reza, et al.** "Membership Inference Attacks Against Machine Learning Models." *IEEE S&P* (2017)
4. **Goodfellow, Ian, et al.** "Generative Adversarial Networks." *NIPS* (2014)

---