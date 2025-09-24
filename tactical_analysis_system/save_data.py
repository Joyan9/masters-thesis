#!/usr/bin/env python3
"""
Data Collection Script with Batch Processing
Collects and saves StatsBomb data for analysis in manageable batches
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tactical_analysis_system.data_loader import DataLoader

def collect_data_batch(batch_size=50, max_matches=None, save_interval=100):
    """Collect and save data in batches"""
    print("COLLECTING STATSBOMB DATA - BATCH MODE")
    print("=" * 50)
    print(f"⚙️  Batch size: {batch_size} matches")
    print(f"⚙️  Save interval: {save_interval} matches" if save_interval else "⚙️  No interim saves")
    print(f"⚙️  Max matches: {max_matches if max_matches else 'All available'}")
    print()
    
    # Define competitions to collect
    competitions = [
        # La Liga
        (11, 90),   # La Liga 2020/2021
        (11, 42),   # La Liga 2019/2020
        # Premier League
        (2, 27),    # Premier League 2015/2016
        # Serie A
        (12, 27),   # Serie A 2015/2016
    ]
    
    # Load data in batches
    print("Loading data from StatsBomb API...")
    dl = DataLoader().load_data(
        competitions, 
        max_matches=max_matches,
        batch_size=batch_size,
        save_interval=save_interval
    )
    
    # Save final data
    output_file = "statsbomb_data.json"
    dl.save_data(output_file)
    
    print(f"\n✅ Data collection complete!")
    print(f"📁 Data saved to: {output_file}")
    print(f"📊 Matches: {len(dl.matches)}")
    print(f"📊 Events: {len(dl.events)} match events")
    
    return output_file

def resume_collection(existing_file="statsbomb_data.json", batch_size=50):
    """Resume data collection from existing file"""
    print("RESUMING STATSBOMB DATA COLLECTION")
    print("=" * 50)
    
    competitions = [
        (11, 90), (11, 42), (2, 27), (12, 27)
    ]
    
    dl = DataLoader().resume_loading(
        competitions,
        existing_data_file=existing_file,
        batch_size=batch_size
    )
    
    # Save updated data
    output_file = "statsbomb_data_resumed.json"
    dl.save_data(output_file)
    
    print(f"\n✅ Resume complete!")
    print(f"📁 Data saved to: {output_file}")
    print(f"📊 Total events: {len(dl.events)}")
    
    return output_file

def collect_small_sample(num_matches=10):
    """Collect a small sample for testing"""
    print(f"COLLECTING SAMPLE DATA - {num_matches} MATCHES")
    print("=" * 50)
    
    competitions = [(2, 27)]  # Just Premier League 2015/2016
    
    dl = DataLoader().load_data(
        competitions, 
        max_matches=num_matches,
        batch_size=5  # Very small batches for testing
    )
    
    output_file = f"statsbomb_sample_{num_matches}.json"
    dl.save_data(output_file)
    
    print(f"\n✅ Sample collection complete!")
    print(f"📁 Data saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect StatsBomb data')
    parser.add_argument('--mode', choices=['full', 'resume', 'sample'], 
                       default='full', help='Collection mode')
    parser.add_argument('--batch-size', type=int, default=50, 
                       help='Number of matches per batch')
    parser.add_argument('--max-matches', type=int, 
                       help='Maximum number of matches to collect')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save interim data every N matches')
    parser.add_argument('--sample-size', type=int, default=10,
                       help='Number of matches for sample mode')
    parser.add_argument('--existing-file', type=str, default="statsbomb_data.json",
                       help='Existing data file for resume mode')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'full':
            collect_data_batch(
                batch_size=args.batch_size, 
                max_matches=args.max_matches,
                save_interval=args.save_interval
            )
        elif args.mode == 'resume':
            resume_collection(
                existing_file=args.existing_file,
                batch_size=args.batch_size
            )
        elif args.mode == 'sample':
            collect_small_sample(args.sample_size)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Collection interrupted by user")
        print("💡 You can resume collection using: python save_data.py --mode resume")
    except Exception as e:
        print(f"\n❌ Error during collection: {e}")
        print("💡 You can try resuming with: python save_data.py --mode resume")