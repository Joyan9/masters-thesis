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
        window_sizes = [10] # reduced to just 10 minute windows
        results_by_window = {}
        network_data_by_window = {}
        for w_size in window_sizes:
            print(f"\n=== Running analysis for window size: {w_size} minutes ===")
            analysis = MainAnalysis(use_saved_data=True, data_file=data_file, window_size=w_size, step_size=w_size//2 or 1)
            print(f"📊 Data summary (window {w_size}):")
            print(f"   - Matches: {len(analysis.data_loader.matches_data)}")
            print(f"   - Events: {len(analysis.data_loader.events_data)}")
            rq1_results = analysis.run_rq1_analysis(max_matches=100, save_results=False, filepath=data_file)
            # Store results for comparison
            results_by_window[w_size] = rq1_results
            # Also store the network data for visualization
            if 'network_data' in rq1_results:
                network_data_by_window[w_size] = rq1_results['network_data']
            # Print summary stats for this window size
            if 'results_df' in rq1_results:
                df = rq1_results['results_df']
                print(f"   - Windows: {len(df)}")
                print(f"   - Mean pass count: {df['pass_count'].mean():.2f}")
                print(f"   - Mean density: {df['density'].mean():.4f}")
            else:
                print("   - No results_df found in RQ1 results.")
        print("\n=== Comparison of window sizes complete ===")

        # Side-by-side visualization for winning vs. losing contexts (using largest window size as example)
        if network_data_by_window:
            from tactical_analysis_system.visualizer import RQ1Visualizer
            largest_window = max(network_data_by_window.keys())
            network_data = network_data_by_window[largest_window]
            print(f"\nGenerating side-by-side passing network visualization for window size {largest_window}...")
            visualizer = RQ1Visualizer()
            visualizer.plot_winning_vs_losing_networks(network_data, save_plot=True, output_filename=f'winning_vs_losing_networks_{largest_window}min.png')
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
