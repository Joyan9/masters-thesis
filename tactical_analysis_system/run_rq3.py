#!/usr/bin/env python3
"""
Run RQ3 Analysis with pre-saved data
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

from tactical_analysis_system.main_analysis import MainAnalysis
from tactical_analysis_system.data_loader import DataLoader
from tactical_analysis_system.visualizer import RQ1Visualizer
import pandas as pd

def main():
    # Check if data file exists
    data_file = "statsbomb_data_interim_100.json"
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        print("Please run data collection first:")
        print("python collect_data.py")
        return
    
    try:
        # Use a single window size for analysis
        window_size = 10
        print(f"\n=== Running analysis for window size: {window_size} minutes ===")
        analysis = MainAnalysis(use_saved_data=True, data_file=data_file, 
                              window_size=window_size, step_size=window_size//2 or 1)
        
        # Initialize visualizer
        visualizer = RQ1Visualizer()
        
        # Run analyses and create visualizations for each RQ
        print("\n1. Running RQ1 Analysis...")
        rq1_results = analysis.run_rq1_analysis(max_matches=100, save_results=True, filepath=data_file)
        
        # Check if required data is available for RQ1 visualization
        if rq1_results and isinstance(rq1_results, dict):
            if 'network_analysis' in rq1_results:
                results_df = pd.DataFrame(rq1_results['network_analysis'])
                statistical_results = rq1_results.get('statistical_results', {})
                
                visualizer.create_context_network_analysis_plots(
                    results_df=results_df,
                    statistical_results=statistical_results,
                    save_plots=True
                )
            else:
                print("⚠️  Warning: RQ1 results missing network analysis data")
        
        print("\n2. Running RQ2 Analysis...")
        rq2_results = analysis.run_rq2_analysis(save_results=True)
        
        # Check if required data is available for RQ2 visualization
        if rq2_results and isinstance(rq2_results, dict):
            recommendations = rq2_results.get('recommendations', [])
            if recommendations:
                visualizer.create_tactical_recommendation_plots(
                    recommendations_data=recommendations,
                    save_plots=True
                )
            else:
                print("⚠️  Warning: RQ2 results missing recommendations data")
        
        print("\n3. Running RQ3 Analysis...")
        rq3_results = analysis.run_rq3_analysis(save_results=True)
        
        # Check if required data is available for RQ3 visualization
        if rq3_results and isinstance(rq3_results, dict):
            counterfactual_results = rq3_results.get('counterfactual_results', {})
            if counterfactual_results:
                visualizer.create_validation_analysis_plots(
                    counterfactual_results=counterfactual_results,
                    save_plots=True
                )
            else:
                print("⚠️  Warning: RQ3 results missing counterfactual analysis data")
        
        print("\n=== Analysis complete ===")
        print("📊 Plots saved in results/plots/")
        print("   - RQ1: Context network analysis")
        print("   - RQ2: Tactical recommendations")
        print("   - RQ3: Validation analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
