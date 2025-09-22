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

    def load_saved_data(self, filename):
        """Load previously saved data"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.matches = data['matches']
        self.events = data['events']
        
        print(f"✅ Data loaded from {filename}")
        print(f"   - Matches: {len(self.matches)}")
        print(f"   - Events: {len(self.events)}")
        print(f"   - Saved on: {data.get('timestamp', 'Unknown')}")
