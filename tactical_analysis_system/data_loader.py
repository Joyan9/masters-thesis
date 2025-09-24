from statsbombpy import sb
import json
from datetime import datetime
import time
import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self):
        self.matches = []
        self.events = {}
        # Initialize pandas DataFrames for compatibility with analysis system
        import pandas as pd
        self.matches_data = pd.DataFrame()
        self.events_data = pd.DataFrame()
        self.lineups_data = pd.DataFrame()
    
    def load_data(self, competitions, max_matches=None, batch_size=50, save_interval=None):
        """Load matches and events data in batches"""
        print("Loading matches and events for context analysis...")
        
        # Load matches first
        for comp_id, season_id in competitions:
            try:
                season_matches = sb.matches(comp_id, season_id)
                self.matches.extend(season_matches.to_dict("records"))
                print(f"✅ Loaded {len(season_matches)} matches for comp {comp_id}, season {season_id}")
            except Exception as e:
                print(f"❌ Error loading comp {comp_id}, season {season_id}: {e}")
        
        # Apply max_matches limit if specified
        match_list = self.matches if max_matches is None else self.matches[:max_matches]
        total_matches = len(match_list)
        print(f"📊 Total matches to process: {total_matches}")
        
        # Process events in batches
        processed_count = 0
        failed_matches = []
        
        for i in range(0, total_matches, batch_size):
            batch_end = min(i + batch_size, total_matches)
            batch_matches = match_list[i:batch_end]
            
            print(f"\n🔄 Processing batch {i//batch_size + 1}: matches {i+1}-{batch_end} of {total_matches}")
            
            batch_success_count = 0
            for match in batch_matches:
                mid = match["match_id"]
                try:
                    self.events[mid] = sb.events(match_id=mid).to_dict("records")
                    batch_success_count += 1
                    processed_count += 1
                    
                    if processed_count % 10 == 0:  # Progress indicator
                        print(f"   ✅ Processed {processed_count}/{total_matches} matches")
                    
                    # Small delay to avoid overwhelming the API
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"   ❌ Failed to load events for match {mid}: {e}")
                    failed_matches.append(mid)
            
            print(f"   Batch complete: {batch_success_count}/{len(batch_matches)} successful")
            
            # Save intermediate results if save_interval is specified
            if save_interval and (i + batch_size) % save_interval == 0:
                interim_filename = f"statsbomb_data_interim_{i + batch_size}.json"
                self._save_interim_data(interim_filename)
                print(f"   💾 Saved interim data: {interim_filename}")
            
            # Brief pause between batches
            time.sleep(0.5)
        
        print(f"\n📊 Data loading complete:")
        print(f"   - Total matches: {len(self.matches)}")
        print(f"   - Successfully loaded events: {len(self.events)}")
        print(f"   - Failed matches: {len(failed_matches)}")
        
        if failed_matches:
            print(f"   - Failed match IDs: {failed_matches[:10]}{'...' if len(failed_matches) > 10 else ''}")
        
        # Create DataFrame versions for compatibility with analysis system
        self._create_dataframes()
        
        return self
    
    def _create_dataframes(self):
        """Create pandas DataFrames from loaded data for analysis compatibility"""
        import pandas as pd
        
        # Create matches_data DataFrame
        if self.matches:
            self.matches_data = pd.DataFrame(self.matches)
            if 'match_date' in self.matches_data.columns:
                self.matches_data['match_date'] = pd.to_datetime(self.matches_data['match_date'])
        else:
            self.matches_data = pd.DataFrame()
        
        # Create events_data DataFrame by combining all events
        if self.events:
            all_events = []
            for match_id, events_list in self.events.items():
                for event in events_list:
                    event['match_id'] = match_id  # Ensure match_id is in each event
                    all_events.append(event)
            
            self.events_data = pd.DataFrame(all_events)
            if 'timestamp' in self.events_data.columns:
                self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
        else:
            self.events_data = pd.DataFrame()
        
        # Create empty lineups_data for compatibility (if you need lineups later)
        if not hasattr(self, 'lineups_data'):
            self.lineups_data = pd.DataFrame()
        
        print(f"   - DataFrames created: {len(self.matches_data)} matches, {len(self.events_data)} events")
    
    def _save_interim_data(self, filename):
        """Save interim data during batch processing"""
        output_data = {
            'matches': self.matches,
            'events': self.events,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_events': len(self.events),
            'status': 'interim'
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

    def save_data(self, filename):
        """Save loaded data"""
        output_data = {
            'matches': self.matches,
            'events': self.events,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_events': len(self.events),
            'status': 'complete'
        }
        
        # Also save DataFrame versions for compatibility
        if hasattr(self, 'matches_data') and not self.matches_data.empty:
            output_data['matches_data'] = self.matches_data.to_dict('records')
        
        if hasattr(self, 'events_data') and not self.events_data.empty:
            output_data['events_data'] = self.events_data.to_dict('records')
        
        if hasattr(self, 'lineups_data'):
            output_data['lineups_data'] = self.lineups_data.to_dict('records')
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Data saved to {filename}")
        print(f"   - Matches: {len(self.matches)}")
        print(f"   - Events: {len(self.events)}")

    def load_from_json(self, filepath: str) -> 'DataLoader':
        """Load previously saved data from JSON file"""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading data from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct DataLoader object
        self.matches = data.get('matches', [])
        self.events = data.get('events', {})
        
        # Convert to pandas DataFrames for compatibility with analysis system
        import pandas as pd
        
        # Create matches_data DataFrame
        if self.matches:
            self.matches_data = pd.DataFrame(self.matches)
            # Convert date column if it exists
            if 'match_date' in self.matches_data.columns:
                self.matches_data['match_date'] = pd.to_datetime(self.matches_data['match_date'])
        else:
            self.matches_data = pd.DataFrame()
        
        # Create events_data DataFrame by combining all events
        if self.events:
            all_events = []
            for match_id, events_list in self.events.items():
                for event in events_list:
                    event['match_id'] = match_id  # Ensure match_id is in each event
                    all_events.extend([event])
            
            self.events_data = pd.DataFrame(all_events)
            # Convert timestamp column if it exists
            if 'timestamp' in self.events_data.columns:
                self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
        else:
            self.events_data = pd.DataFrame()
        
        # Create empty lineups_data for compatibility
        self.lineups_data = pd.DataFrame()
        
        print(f"✅ Loaded {len(self.matches)} matches, {len(self.events)} match events")
        print(f"✅ Created DataFrames: {len(self.matches_data)} matches, {len(self.events_data)} events")
        
        return self
    
    def resume_loading(self, competitions, existing_data_file=None, batch_size=50, max_matches=None, save_interval=100):
        """Resume loading from where it left off"""
        print("🔄 Resuming data loading...")
        
        # Load existing data if provided
        if existing_data_file and Path(existing_data_file).exists():
            self.load_from_json(existing_data_file)
            print(f"📁 Loaded existing data: {len(self.events)} events already collected")
        
        # Load matches if not already loaded
        if not self.matches:
            for comp_id, season_id in competitions:
                try:
                    season_matches = sb.matches(comp_id, season_id)
                    self.matches.extend(season_matches.to_dict("records"))
                    print(f"✅ Loaded {len(season_matches)} matches for comp {comp_id}, season {season_id}")
                except Exception as e:
                    print(f"❌ Error loading comp {comp_id}, season {season_id}: {e}")
        
        # Find matches that need events loaded
        match_list = self.matches if max_matches is None else self.matches[:max_matches]
        remaining_matches = [match for match in match_list 
                           if str(match["match_id"]) not in self.events]
        
        total_collected = len(self.events)
        total_remaining = len(remaining_matches)
        total_matches = len(match_list)
        
        print(f"📊 Progress status:")
        print(f"   - Already collected: {total_collected}/{total_matches} matches")
        print(f"   - Remaining to process: {total_remaining} matches")
        
        if remaining_matches:
            # Continue with batch processing for remaining matches
            processed_in_resume = 0
            
            for i in range(0, len(remaining_matches), batch_size):
                batch_end = min(i + batch_size, len(remaining_matches))
                batch_matches = remaining_matches[i:batch_end]
                
                current_total = total_collected + processed_in_resume
                print(f"\n🔄 Processing batch: matches {current_total + 1}-{current_total + len(batch_matches)} of {total_matches}")
                
                batch_success_count = 0
                for match in batch_matches:
                    mid = match["match_id"]
                    try:
                        self.events[mid] = sb.events(match_id=mid).to_dict("records")
                        batch_success_count += 1
                        processed_in_resume += 1
                        
                        if (total_collected + processed_in_resume) % 10 == 0:
                            print(f"   ✅ Total processed: {total_collected + processed_in_resume}/{total_matches} matches")
                        
                        time.sleep(0.1)  # Small delay
                    except Exception as e:
                        print(f"   ❌ Failed to load events for match {mid}: {e}")
                
                print(f"   Batch complete: {batch_success_count}/{len(batch_matches)} successful")
                
                # Save interim data during resume if specified
                if save_interval and (processed_in_resume % save_interval == 0):
                    interim_filename = f"statsbomb_data_resumed_{total_collected + processed_in_resume}.json"
                    self._save_interim_data(interim_filename)
                    print(f"   💾 Saved interim data: {interim_filename}")
                
                # Brief pause between batches
                time.sleep(0.5)
        
        final_total = len(self.events)
        print(f"✅ Resume complete: {final_total} total events loaded")
        print(f"   - New events collected: {final_total - total_collected}")
        return self