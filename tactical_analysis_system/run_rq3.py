#!/usr/bin/env python3

import sys
from pathlib import Path
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint

from tactical_analysis_system.main_analysis import MainAnalysis
from tactical_analysis_system.data_loader import DataLoader
from tactical_analysis_system.visualizer import RQ1Visualizer

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 1. Overall Counterfactual Impact Summary
def plot_treatment_effect_distribution(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    comparisons = rq3_results['counterfactual_results']['comparison_results']['individual_comparisons']
    metrics = list(comparisons[0]['metric_comparisons'].keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        tau = [comp['metric_comparisons'][metric]['difference'] for comp in comparisons if metric in comp['metric_comparisons']]
        mean_tau = np.mean(tau)
        ci = bootstrap((np.array(tau),), np.mean, confidence_level=0.95, n_resamples=1000).confidence_interval
        sns.histplot(tau, bins=20, ax=axes[i], kde=True, color='dodgerblue', alpha=0.7)
        axes[i].axvline(0, color='black', linestyle='--', label='Null boundary')
        axes[i].axvline(mean_tau, color='red', linestyle='-', label=f'Mean τ={mean_tau:.3f}')
        axes[i].fill_betweenx([0, axes[i].get_ylim()[1]], ci.low, ci.high, color='gray', alpha=0.2, label='95% CI')
        axes[i].set_title(f"{metric.replace('_',' ').title()} Treatment Effect")
        axes[i].set_xlabel("τ = Counterfactual - Observed")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure1_treatment_effect_distribution.png", dpi=300)
    plt.close()

# 2. Improvement Rate Analysis
def plot_recommendation_success_rates(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    summary = rq3_results['counterfactual_results']['comparison_results']['summary_statistics']
    rates = summary['improvement_rates']
    metrics = list(rates.keys())
    values = [rates[m] for m in metrics]
    n = rq3_results['counterfactual_results']['comparison_results']['total_comparisons']
    # Wilson score intervals
    intervals = [
    proportion_confint(count=int(values[i] * n), nobs=n, alpha=0.05, method='wilson')
    for i in range(len(metrics))]
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, yerr=[(hi-lo)/2 for lo, hi in intervals], color='mediumseagreen', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle='--', label='Null hypothesis rate')
    plt.ylabel("Improvement Rate (τ > 0)")
    plt.title("Figure 2: Recommendation Success Rates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure2_recommendation_success_rates.png", dpi=300)
    plt.close()

# 3. Treatment Effect by Recommendation Type
def plot_stratified_treatment_effects(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    impacts = rq3_results['counterfactual_results']['impact_analysis']['recommendation_type_impacts']
    metrics = list(rq3_results['counterfactual_results']['comparison_results']['summary_statistics']['improvement_rates'].keys())
    rec_types = list(impacts.keys())
    data = []
    for rec_type in rec_types:
        for metric in metrics:
            rate = impacts[rec_type].get('improvement_rate', np.nan)
            data.append({'Recommendation Type': rec_type, 'Metric': metric, 'Improvement Rate': rate})
    df = pd.DataFrame(data)
    g = sns.catplot(data=df, x='Recommendation Type', y='Improvement Rate', col='Metric', kind='bar', col_wrap=2, palette='Set2', height=4)
    g.fig.suptitle("Figure 3: Stratified Treatment Effects by Recommendation Type", y=1.03)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure3_stratified_treatment_effects.png", dpi=300)
    plt.close()

# 4. Counterfactual Scenario Examples
def plot_counterfactual_case_studies(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    # This is a template; you must adapt for your actual network/trajectory plotting
    case_studies = rq3_results['counterfactual_results'].get('scenarios', [])[:3]
    fig, axes = plt.subplots(len(case_studies), 3, figsize=(15, 5*len(case_studies)))
    for i, case in enumerate(case_studies):
        # Left: observed network state (placeholder)
        axes[i,0].set_title("Observed Network")
        axes[i,0].text(0.5, 0.5, "Network plot here", ha='center', va='center')
        axes[i,0].axis('off')
        # Middle: recommendation issued
        rec_types = [r['type'] for r in case.get('recommendations',[])]
        axes[i,1].set_title("Recommendation Issued")
        axes[i,1].text(0.5, 0.5, ", ".join(rec_types), ha='center', va='center')
        axes[i,1].axis('off')
        # Right: trajectory comparison (placeholder)
        axes[i,2].set_title("Trajectory Comparison")
        axes[i,2].plot([0,1,2], [case['actual_metrics'].get('density',0), case['actual_metrics'].get('density',0)+0.03, case['actual_metrics'].get('density',0)+0.12], label='Counterfactual', color='green')
        axes[i,2].plot([0,1,2], [case['actual_metrics'].get('density',0), case['actual_metrics'].get('density',0)+0.01, case['actual_metrics'].get('density',0)+0.03], label='Observed', color='blue')
        axes[i,2].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure4_counterfactual_case_studies.png", dpi=300)
    plt.close()

# 5. Validation Score Breakdown
def plot_validation_score_breakdown(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    validation = rq3_results['counterfactual_results']
    # Use impact_analysis and model_performance for components
    components = [
        ('Historical Correlation', validation.get('impact_analysis', {}).get('confidence_correlation', 0)),
        ('Improvement Rate', validation.get('impact_analysis', {}).get('overall_improvement_rate', 0)),
        ('Treatment Effect Magnitude', np.mean([v for v in validation.get('impact_analysis', {}).get('metric_impacts', {}).values()])),
        ('Temporal Consistency', validation.get('model_performance', {}).get('overall_quality', 0))
    ]
    labels, values = zip(*components)
    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color=['#3498db','#2ecc71','#e67e22','#9b59b6'])
    plt.ylim(0,1)
    plt.title("Figure 5: Validation Score Breakdown")
    plt.ylabel("Component Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure5_validation_score_breakdown.png", dpi=300)
    plt.close()

# 6. Comparison with Baseline/Null Models
def plot_treatment_effect_vs_null(rq3_results, output_dir="results/plots/rq3"):
    ensure_dir(output_dir)
    # You must provide null distributions in your results for this plot
    # Here we use a placeholder for null distribution
    comparisons = rq3_results['counterfactual_results']['comparison_results']['individual_comparisons']
    metrics = list(comparisons[0]['metric_comparisons'].keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    for i, metric in enumerate(metrics):
        tau = [comp['metric_comparisons'][metric]['difference'] for comp in comparisons if metric in comp['metric_comparisons']]
        # Placeholder for null: random normal with same mean/var
        null_tau = np.random.normal(np.mean(tau), np.std(tau), size=1000)
        sns.histplot(null_tau, bins=20, ax=axes[i], color='gray', alpha=0.5, label='Null')
        sns.histplot(tau, bins=20, ax=axes[i], color='red', alpha=0.7, label='Observed')
        axes[i].axvline(np.mean(null_tau), color='gray', linestyle='--', label='Null Mean')
        axes[i].axvline(np.mean(tau), color='red', linestyle='-', label='Observed Mean')
        axes[i].set_title(f"{metric.replace('_',' ').title()} Treatment Effect vs Null")
        axes[i].set_xlabel("τ")
        axes[i].legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figure6_treatment_effect_vs_null.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    # Check if data file exists
    data_file = "statsbomb_data_interim_100.json"
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
   
    try:
        # Use a single window size for analysis
        window_size = 10
        print(f"\n=== Running analysis for window size: {window_size} minutes ===")
        analysis = MainAnalysis(use_saved_data=True, data_file=data_file, 
                              window_size=window_size, step_size=window_size//2 or 1)
        
        # Run analyses and create visualizations for each RQ
        print("\n1. Running RQ1 Analysis...")
        rq1_results = analysis.run_rq1_analysis(max_matches=100, save_results=True, filepath=data_file)
                
        print("\n2. Running RQ2 Analysis...")
        rq2_results = analysis.run_rq2_analysis(save_results=True)
        
        print("\n3. Running RQ3 Analysis...")
        rq3_results = analysis.run_rq3_analysis(save_results=True)
        
        output_dir = "/workspaces/masters-thesis/results/plots/rq3"
        print("\n=== Running plot_treatment_effect_distribution ===")
        plot_treatment_effect_distribution(rq3_results, output_dir)

        print("\n=== Running plot_recommendation_success_rates ===")
        plot_recommendation_success_rates(rq3_results, output_dir)

        print("\n=== Running plot_stratified_treatment_effects ===")
        plot_stratified_treatment_effects(rq3_results, output_dir)

        print("\n=== Running plot_counterfactual_case_studies ===")
        plot_counterfactual_case_studies(rq3_results, output_dir)

        print("\n=== Running plot_validation_score_breakdown ===")
        plot_validation_score_breakdown(rq3_results, output_dir)

        print("\n=== Running plot_treatment_effect_vs_null ===")
        plot_treatment_effect_vs_null(rq3_results, output_dir)

        print("\n=== Analysis complete ===")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
