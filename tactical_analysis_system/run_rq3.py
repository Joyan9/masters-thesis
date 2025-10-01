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
        analysis = MainAnalysis(use_saved_data=True, data_file=data_file, window_size=window_size, step_size=window_size//2 or 1)
        print(f"📊 Data summary (window {window_size}):")
        print(f"   - Matches: {len(analysis.data_loader.matches_data)}")
        print(f"   - Events: {len(analysis.data_loader.events_data)}")
        rq1_results = analysis.run_rq1_analysis(max_matches=100, save_results=False, filepath=data_file)
        # Print summary stats for this window size
        if 'results_df' in rq1_results:
            df = rq1_results['results_df']
            print(f"   - Windows: {len(df)}")
            print(f"   - Mean pass count: {df['pass_count'].mean():.2f}")
            print(f"   - Mean density: {df['density'].mean():.4f}")
        else:
            print("   - No results_df found in RQ1 results.")

        # Side-by-side visualization for winning vs. losing contexts
        if 'network_data' in rq1_results:
            from tactical_analysis_system.visualizer import RQ1Visualizer
            network_data = rq1_results['network_data']
            print(f"\nGenerating side-by-side passing network visualization for window size {window_size}...")
            visualizer = RQ1Visualizer()
            visualizer.plot_winning_vs_losing_networks(network_data, save_plot=True, output_filename=f'winning_vs_losing_networks_{window_size}min.png')
        
        # Run RQ2 and RQ3 analyses and save reports
        rq2_results = analysis.run_rq2_analysis(save_results=True)
        rq3_results = analysis.run_rq3_analysis(save_results=True)

        print("\n=== RQ3 analysis complete ===")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
