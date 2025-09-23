#!/usr/bin/env python3
"""
Run RQ3 Analysis with pre-saved data
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tactical_analysis_system.main_analysis import TacticalAnalysisSystem

def main():
    # Check if data file exists
    data_file = "statsbomb_data.json"
    if not Path(data_file).exists():
        print(f"❌ Data file {data_file} not found!")
        print("Please run data collection first:")
        print("python collect_data.py")
        return
    
    # Initialize analysis system with saved data
    analysis = TacticalAnalysisSystem(use_saved_data=True, data_file=data_file)
    
    # Run analyses
    print("Running RQ1 Analysis...")
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

if __name__ == "__main__":
    main()
