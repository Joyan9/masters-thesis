from statsbombpy import sb
import json
from datetime import datetime


class DataLoader:
    def __init__(self):
        self.matches = []
        self.events = {}
    
    def load_data(self, competitions, max_matches=None):
        """Load matches and events data"""
        print("Loading matches and events for context analysis...")
        
        # Load matches
        for comp_id, season_id in competitions:
            try:
                season_matches = sb.matches(comp_id, season_id)
                self.matches.extend(season_matches.to_dict("records"))
                print(f"✅ Loaded {len(season_matches)} matches for comp {comp_id}, season {season_id}")
            except Exception as e:
                print(f"❌ Error loading comp {comp_id}, season {season_id}: {e}")
        
        # Load events
        match_list = self.matches if max_matches is None else self.matches[:max_matches]
        for match in match_list:
            mid = match["match_id"]
            try:
                self.events[mid] = sb.events(match_id=mid).to_dict("records")
                print(f"✅ Loaded events for match {mid}")
            except Exception as e:
                print(f"❌ Could not load events for match {mid}: {e}")
        
        print(f"Data loaded: {len(self.matches)} matches, {len(self.events)} match events")
        return self

    def save_data(self, filename):
        """Save loaded data"""
        output_data = {
            'matches': self.matches,
            'events': self.events,
            'timestamp': datetime.now().isoformat(),
            'total_matches': len(self.matches),
            'total_events': len(self.events)
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Data saved to {filename}")
        print(f"   - Matches: {len(self.matches)}")
        print(f"   - Events: {len(self.events)}")

    def load_from_json(self, filepath: str) -> 'DataLoader':
        """Load previously saved data from JSON file"""
        import json
        from pathlib import Path
        
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"Loading data from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Reconstruct DataLoader object
        self.matches_data = pd.DataFrame(data['matches_data']) if data['matches_data'] else pd.DataFrame()
        self.events_data = pd.DataFrame(data['events_data']) if data['events_data'] else pd.DataFrame()
        self.lineups_data = pd.DataFrame(data['lineups_data']) if data['lineups_data'] else pd.DataFrame()
        
        # Convert datetime columns back
        if not self.matches_data.empty and 'match_date' in self.matches_data.columns:
            self.matches_data['match_date'] = pd.to_datetime(self.matches_data['match_date'])
        
        if not self.events_data.empty and 'timestamp' in self.events_data.columns:
            self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'])
        
        print(f"✅ Loaded {len(self.matches_data)} matches, {len(self.events_data)} events, {len(self.lineups_data)} lineups")
        
        return self

