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
        print("\nInitializing analysis system with saved data...")
        # Create MainAnalysis with the saved data file
        analysis = MainAnalysis(use_saved_data=True, data_file=data_file)
        
        print(f"📊 Data summary:")
        print(f"   - Matches: {len(analysis.data_loader.matches_data)}")
        print(f"   - Events: {len(analysis.data_loader.events_data)}")
        print(f"   - Lineups: {len(analysis.data_loader.lineups_data)}")
        
        # Run analyses
        print("\nRunning RQ1 Analysis...")
        rq1_results = analysis.run_rq1_analysis(save_results=True)
        
        print("\nRunning RQ2 Analysis...")
        rq2_results = analysis.run_rq2_analysis(save_results=True)
        
        print("\nRunning RQ3 Analysis...")
        rq3_results = analysis.run_rq3_analysis(save_results=True)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"RQ1 Score: {rq1_results.get('overall_score', 'N/A')}")
        print(f"RQ2 Score: {rq2_results.get('overall_score', 'N/A')}")
        print(f"RQ3 Score: {rq3_results.get('overall_score', 'N/A')}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
