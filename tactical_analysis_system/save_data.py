#!/usr/bin/env python3
"""
Data Collection Script
Collects and saves StatsBomb data for analysis
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tactical_analysis_system.data_loader import DataLoader

def collect_data():
    """Collect and save data"""
    print("COLLECTING STATSBOMB DATA")
    print("=" * 50)
    
    # Define competitions to collect
    competitions=[
                (2, 27),    # Premier League 2015/2016
                #(12, 27)  # Serie A 2015/2016
            ]
    
    # Load data
    print("Loading data from StatsBomb API...")
    dl = DataLoader().load_data(competitions, max_matches=None)
    
    # Save data
    output_file = "statsbomb_data.json"
    dl.save_data(output_file)
    
    print(f"\n✅ Data collection complete!")
    print(f"📁 Data saved to: {output_file}")
    print(f"📊 Matches: {len(dl.matches_data)}")
    print(f"📊 Events: {len(dl.events_data)}")
    print(f"📊 Lineups: {len(dl.lineups_data)}")
    
    return output_file

if __name__ == "__main__":
    collect_data()
