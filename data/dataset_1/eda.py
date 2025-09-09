import json
import zipfile
import os
import pandas as pd
from collections import defaultdict
import numpy as np

class FootballDatasetExplorer:
    def __init__(self, data_path="data/dataset_1/"):
        self.data_path = data_path
        self.data = {}
        self.events_data = {}
        
    def load_json_files(self):
        """Load all JSON files in the dataset with error handling"""
        json_files = [
            'coaches.json',
            'competitions.json', 
            'players.json',
            'referees.json',
            'teams.json'
        ]
        
        # Load regular JSON files
        for file in json_files:
            file_path = os.path.join(self.data_path, file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.data[file.replace('.json', '')] = json.load(f)
                    print(f"✅ Loaded {file}: {len(self.data[file.replace('.json', '')])} records")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Error in {file}: {e}")
                    print(f"   Error at line {e.lineno}, column {e.colno}")
                    self._try_fix_json_file(file_path, file)
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
        
        # Load matches files
        matches_files = [
            'matches_England.json',
            'matches_European_Championship.json',
            'matches_France.json',
            'matches_Germany.json',
            'matches_Italy.json',
            'matches_Spain.json',
            'matches_World_Cup.json'
        ]
        
        self.data['matches'] = []
        for file in matches_files:
            file_path = os.path.join(self.data_path, 'matches', file)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        matches = json.load(f)
                        self.data['matches'].extend(matches)
                        print(f"✅ Loaded {file}: {len(matches)} matches")
                except json.JSONDecodeError as e:
                    print(f"❌ JSON Error in {file}: {e}")
                    print(f"   Error at line {e.lineno}, column {e.colno}")
                    self._try_fix_json_file(file_path, file)
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
        
        print(f"Total matches loaded: {len(self.data['matches'])}")
    
    def _try_fix_json_file(self, file_path, file_name):
        """Try to diagnose and potentially fix JSON issues"""
        print(f"🔧 Attempting to diagnose {file_name}...")
        
        try:
            # Read file and check for common issues
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check file size
            print(f"   File size: {len(content):,} characters")
            
            # Look for common JSON issues
            if content.count('[') != content.count(']'):
                print(f"   ⚠️ Mismatched brackets: [ count={content.count('[')} ] count={content.count(']')}")
            
            if content.count('{') != content.count('}'):
                print(f"   ⚠️ Mismatched braces: {{ count={content.count('{')} }} count={content.count('}')}")
            
            # Check for trailing commas or other issues around the error position
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                error_pos = e.pos
                start = max(0, error_pos - 100)
                end = min(len(content), error_pos + 100)
                context = content[start:end]
                print(f"   Context around error position {error_pos}:")
                print(f"   '{context}'")
                
                # Try to load as JSONL (one JSON object per line)
                print("   🔄 Trying to parse as JSONL (newline-delimited JSON)...")
                self._try_load_jsonl(file_path, file_name)
                
        except Exception as e:
            print(f"   ❌ Could not diagnose file: {e}")
    
    def _try_load_jsonl(self, file_path, file_name):
        """Try to load file as JSONL (newline-delimited JSON)"""
        try:
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"   ❌ JSONL error at line {line_num}: {e}")
                            break
            
            if records:
                key = file_name.replace('.json', '')
                if key == 'matches' and 'matches' in self.data:
                    self.data['matches'].extend(records)
                else:
                    self.data[key] = records
                print(f"   ✅ Successfully loaded as JSONL: {len(records)} records")
                return True
                
        except Exception as e:
            print(f"   ❌ JSONL loading failed: {e}")
        
        return False
        
    def load_events_from_zip(self):
        """Load events from zip file with error handling"""
        zip_path = os.path.join(self.data_path, 'events.zip')
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_file:
                    for file_name in zip_file.namelist():
                        if file_name.endswith('.json'):
                            try:
                                with zip_file.open(file_name) as f:
                                    events = json.load(f)
                                    country = file_name.replace('events_', '').replace('.json', '')
                                    self.events_data[country] = events
                                    print(f"✅ Loaded {file_name}: {len(events)} events")
                            except json.JSONDecodeError as e:
                                print(f"❌ JSON Error in {file_name}: {e}")
                                print(f"   Skipping this events file...")
                            except Exception as e:
                                print(f"❌ Error loading {file_name}: {e}")
            except Exception as e:
                print(f"❌ Error reading events.zip: {e}")
        else:
            print("❌ events.zip not found")
        
        # Combine all events
        all_events = []
        for country, events in self.events_data.items():
            all_events.extend(events)
        self.data['events'] = all_events
        print(f"Total events loaded: {len(all_events)}")
        
    def explore_data_structure(self):
        """Explore the structure of each dataset"""
        print("\n" + "="*50)
        print("DATA STRUCTURE EXPLORATION")
        print("="*50)
        
        for name, dataset in self.data.items():
            print(f"\n📊 {name.upper()}")
            print("-" * 30)
            
            if not dataset:
                print("❌ No data found")
                continue
                
            print(f"Records: {len(dataset)}")
            
            # Show sample record structure
            sample = dataset[0]
            print(f"Sample record structure:")
            self._print_structure(sample, indent=2)
            
    def _print_structure(self, obj, indent=0, max_depth=3):
        """Recursively print object structure"""
        if indent > max_depth * 2:
            print("  " * indent + "...")
            return
            
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)) and value:
                    print("  " * indent + f"{key}: {type(value).__name__}")
                    if isinstance(value, list):
                        print("  " * indent + f"  └─ [{len(value)} items]")
                        if value:
                            self._print_structure(value[0], indent + 2, max_depth)
                    else:
                        self._print_structure(value, indent + 1, max_depth)
                else:
                    print("  " * indent + f"{key}: {value}")
        elif isinstance(obj, list) and obj:
            print("  " * indent + f"[{len(obj)} items] - Sample:")
            self._print_structure(obj[0], indent + 1, max_depth)
            
    def analyze_relationships(self):
        """Analyze relationships between datasets"""
        print("\n" + "="*50)
        print("RELATIONSHIP ANALYSIS")
        print("="*50)
        
        # Extract unique IDs from each dataset
        ids = {}
        
        # Teams
        if 'teams' in self.data:
            ids['team_ids'] = set(team['wyId'] for team in self.data['teams'])
            print(f"🏟️  Unique teams: {len(ids['team_ids'])}")
            
        # Players
        if 'players' in self.data:
            ids['player_ids'] = set(player['wyId'] for player in self.data['players'])
            print(f"👤 Unique players: {len(ids['player_ids'])}")
            
        # Coaches
        if 'coaches' in self.data:
            ids['coach_ids'] = set(coach['wyId'] for coach in self.data['coaches'])
            print(f"👨‍💼 Unique coaches: {len(ids['coach_ids'])}")
            
        # Referees
        if 'referees' in self.data:
            ids['referee_ids'] = set(referee['wyId'] for referee in self.data['referees'])
            print(f"👨‍⚖️ Unique referees: {len(ids['referee_ids'])}")
            
        # Competitions
        if 'competitions' in self.data:
            ids['competition_ids'] = set(comp['wyId'] for comp in self.data['competitions'])
            print(f"🏆 Unique competitions: {len(ids['competition_ids'])}")
            
        # Matches
        if 'matches' in self.data:
            ids['match_ids'] = set(match['wyId'] for match in self.data['matches'])
            print(f"⚽ Unique matches: {len(ids['match_ids'])}")
            
        print("\n🔗 RELATIONSHIP VALIDATION:")
        
        # Check match relationships
        if 'matches' in self.data:
            self._validate_match_relationships(ids)
            
        # Check events relationships
        if 'events' in self.data:
            self._validate_events_relationships(ids)
            
    def _validate_match_relationships(self, ids):
        """Validate relationships in match data"""
        print("\n📋 Match Data Relationships:")
        
        # Track referenced IDs
        referenced_teams = set()
        referenced_coaches = set()
        referenced_referees = set()
        referenced_competitions = set()
        referenced_players = set()
        
        for match in self.data['matches']:
            # Competition
            if 'competitionId' in match:
                referenced_competitions.add(match['competitionId'])
                
            # Teams and coaches in teamsData
            if 'teamsData' in match:
                for team_id, team_data in match['teamsData'].items():
                    referenced_teams.add(int(team_id))
                    if 'coachId' in team_data:
                        referenced_coaches.add(team_data['coachId'])
                        
                    # Players in formations
                    if 'formation' in team_data:
                        formation = team_data['formation']
                        for player in formation.get('lineup', []):
                            referenced_players.add(player['playerId'])
                        for player in formation.get('bench', []):
                            referenced_players.add(player['playerId'])
                        for sub in formation.get('substitutions', []):
                            referenced_players.add(sub['playerIn'])
                            referenced_players.add(sub['playerOut'])
                            
            # Referees
            if 'referees' in match:
                for ref in match['referees']:
                    referenced_referees.add(ref['refereeId'])
        
        # Validate relationships
        if 'team_ids' in ids:
            missing_teams = referenced_teams - ids['team_ids']
            print(f"  Teams: {len(referenced_teams - missing_teams)}/{len(referenced_teams)} found in teams.json")
            if missing_teams:
                print(f"    ❌ Missing teams: {len(missing_teams)}")
                
        if 'coach_ids' in ids:
            missing_coaches = referenced_coaches - ids['coach_ids']
            print(f"  Coaches: {len(referenced_coaches - missing_coaches)}/{len(referenced_coaches)} found in coaches.json")
            if missing_coaches:
                print(f"    ❌ Missing coaches: {len(missing_coaches)}")
                
        if 'referee_ids' in ids:
            missing_referees = referenced_referees - ids['referee_ids']
            print(f"  Referees: {len(referenced_referees - missing_referees)}/{len(referenced_referees)} found in referees.json")
            if missing_referees:
                print(f"    ❌ Missing referees: {len(missing_referees)}")
                
        if 'competition_ids' in ids:
            missing_competitions = referenced_competitions - ids['competition_ids']
            print(f"  Competitions: {len(referenced_competitions - missing_competitions)}/{len(referenced_competitions)} found in competitions.json")
            if missing_competitions:
                print(f"    ❌ Missing competitions: {len(missing_competitions)}")
                
        if 'player_ids' in ids:
            missing_players = referenced_players - ids['player_ids']
            print(f"  Players: {len(referenced_players - missing_players)}/{len(referenced_players)} found in players.json")
            if missing_players:
                print(f"    ❌ Missing players: {len(missing_players)}")
        
    def _validate_events_relationships(self, ids):
        """Validate relationships in events data"""
        print("\n📊 Events Data Relationships:")
        
        # Sample first 1000 events for performance
        sample_events = self.data['events'][:1000] if len(self.data['events']) > 1000 else self.data['events']
        
        referenced_matches = set()
        referenced_teams = set()
        referenced_players = set()
        
        for event in sample_events:
            if 'matchId' in event:
                referenced_matches.add(event['matchId'])
            if 'teamId' in event:
                referenced_teams.add(event['teamId'])
            if 'playerId' in event:
                referenced_players.add(event['playerId'])
                
        print(f"  Sample size: {len(sample_events)} events")
        
        if 'match_ids' in ids:
            found_matches = referenced_matches & ids['match_ids']
            print(f"  Matches: {len(found_matches)}/{len(referenced_matches)} found in match data")
            
        if 'team_ids' in ids:
            found_teams = referenced_teams & ids['team_ids']
            print(f"  Teams: {len(found_teams)}/{len(referenced_teams)} found in teams.json")
            
        if 'player_ids' in ids:
            found_players = referenced_players & ids['player_ids']
            print(f"  Players: {len(found_players)}/{len(referenced_players)} found in players.json")
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        print("\n" + "="*50)
        print("DATASET SUMMARY STATISTICS")
        print("="*50)
        
        for name, dataset in self.data.items():
            if not dataset:
                continue
                
            print(f"\n📈 {name.upper()}")
            print("-" * 20)
            print(f"Total records: {len(dataset):,}")
            
            if name == 'matches':
                self._match_statistics()
            elif name == 'events':
                self._event_statistics()
            elif name == 'players':
                self._player_statistics()
                
    def _match_statistics(self):
        """Generate match-specific statistics"""
        matches = self.data['matches']
        
        # Count by competition
        competitions = defaultdict(int)
        for match in matches:
            if 'competitionId' in match:
                competitions[match['competitionId']] += 1
                
        print(f"Competitions represented: {len(competitions)}")
        
        # Count by status
        statuses = defaultdict(int)
        for match in matches:
            if 'status' in match:
                statuses[match['status']] += 1
                
        print("Match statuses:")
        for status, count in statuses.items():
            print(f"  {status}: {count}")
            
    def _event_statistics(self):
        """Generate event-specific statistics"""
        events = self.data['events'][:10000]  # Sample for performance
        
        # Count by event type
        event_types = defaultdict(int)
        for event in events:
            if 'eventName' in event:
                event_types[event['eventName']] += 1
                
        print(f"Event types (top 10):")
        sorted_events = sorted(event_types.items(), key=lambda x: x[1], reverse=True)
        for event_type, count in sorted_events[:10]:
            print(f"  {event_type}: {count}")
            
    def _player_statistics(self):
        """Generate player-specific statistics"""
        players = self.data['players']
        
        # Count by role
        roles = defaultdict(int)
        for player in players:
            if 'role' in player and 'name' in player['role']:
                roles[player['role']['name']] += 1
                
        print("Player positions:")
        for role, count in sorted(roles.items(), key=lambda x: x[1], reverse=True):
            print(f"  {role}: {count}")
            
    def run_full_exploration(self):
        """Run complete dataset exploration"""
        print("🚀 Starting Football Dataset Exploration...")
        print("="*50)
        
        # Load all data
        self.load_json_files()
        self.load_events_from_zip()
        
        # Explore structure
        self.explore_data_structure()
        
        # Analyze relationships
        self.analyze_relationships()
        
        # Generate statistics
        self.generate_summary_stats()
        
        print("\n✅ Exploration complete!")

# Usage example
if __name__ == "__main__":
    explorer = FootballDatasetExplorer()
    explorer.run_full_exploration()
    
    # Additional specific queries you can run:
    
    # Get specific team info
    def find_team(team_name):
        teams = explorer.data.get('teams', [])
        return [team for team in teams if team_name.lower() in team['name'].lower()]
    
    # Get matches for a specific competition
    def get_matches_by_competition(competition_id):
        matches = explorer.data.get('matches', [])
        return [match for match in matches if match.get('competitionId') == competition_id]
    
    # Example usage:
    manchester_teams = find_team("Manchester")
    print(f"Found Manchester teams: {len(manchester_teams)}")