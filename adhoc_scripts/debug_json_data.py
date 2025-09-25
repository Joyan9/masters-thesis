import json
from pathlib import Path

# Load and inspect the JSON structure
data_file = "statsbomb_data_interim_100.json"
with open(data_file, 'r') as f:
    data = json.load(f)

print("JSON Data Structure:")
print(f"Top-level keys: {list(data.keys())}")

if 'matches' in data and data['matches']:
    print(f"\nMatches structure:")
    print(f"- Total matches: {len(data['matches'])}")
    print(f"- Sample match keys: {list(data['matches'][0].keys())}")
    print(f"- Sample match: {data['matches'][0]}")

if 'events' in data and data['events']:
    print(f"\nEvents structure:")
    print(f"- Total match IDs with events: {len(data['events'])}")
    
    # Get first match with events
    first_match_id = list(data['events'].keys())[0]
    first_match_events = data['events'][first_match_id]
    
    print(f"- Sample match ID: {first_match_id}")
    print(f"- Events in sample match: {len(first_match_events)}")
    
    if first_match_events:
        print(f"- Sample event keys: {list(first_match_events[0].keys())}")
        print(f"- Sample event: {first_match_events[0]}")
        
        # Check for different possible column names
        sample_event = first_match_events[0]
        possible_type_cols = [k for k in sample_event.keys() if 'type' in k.lower()]
        possible_minute_cols = [k for k in sample_event.keys() if 'minute' in k.lower() or 'time' in k.lower()]
        possible_team_cols = [k for k in sample_event.keys() if 'team' in k.lower()]
        
        print(f"- Possible type columns: {possible_type_cols}")
        print(f"- Possible minute/time columns: {possible_minute_cols}")
        print(f"- Possible team columns: {possible_team_cols}")
