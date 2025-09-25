#!/usr/bin/env python3
"""
Data Description Script
Analyzes and describes the StatsBomb data in your JSON format
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter
from datetime import datetime

def describe_data(json_file: str = "statsbomb_data_interim_100.json"):
    """Describe the data in the JSON file"""
    
    if not Path(json_file).exists():
        print(f"❌ Data file {json_file} not found!")
        return
    
    print("STATSBOMB DATA DESCRIPTION")
    print("=" * 50)
    
    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"📁 Data file: {json_file}")
    print(f"📊 File size: {Path(json_file).stat().st_size / (1024*1024):.1f} MB")
    print(f"🔑 Top-level keys: {list(data.keys())}")
    print()
    
    # Analyze matches
    if 'matches' in data and data['matches']:
        matches = data['matches']
        print("MATCHES DATA")
        print("-" * 30)
        print(f"Total matches: {len(matches)}")
        
        # Convert to DataFrame for easier analysis
        matches_df = pd.DataFrame(matches)
        
        # Competitions
        if 'competition' in matches_df.columns:
            competitions = matches_df['competition'].apply(lambda x: x.get('competition_name', 'Unknown') if isinstance(x, dict) else str(x))
            comp_counts = competitions.value_counts()
            print("\nCompetitions:")
            for comp, count in comp_counts.items():
                print(f"• {comp}: {count} matches")
        
        # Seasons
        if 'season' in matches_df.columns:
            seasons = matches_df['season'].apply(lambda x: x.get('season_name', 'Unknown') if isinstance(x, dict) else str(x))
            season_counts = seasons.value_counts()
            print("\nSeasons:")
            for season, count in season_counts.items():
                print(f"• {season}: {count} matches")
        
        # Teams
        print("\nTeams:")
        teams = set()
        if 'home_team' in matches_df.columns and 'away_team' in matches_df.columns:
            home_teams = matches_df['home_team'].apply(lambda x: x.get('home_team_name', 'Unknown') if isinstance(x, dict) else str(x))
            away_teams = matches_df['away_team'].apply(lambda x: x.get('away_team_name', 'Unknown') if isinstance(x, dict) else str(x))
            
            teams.update(home_teams.dropna())
            teams.update(away_teams.dropna())
            print(f"Total unique teams: {len(teams)}")
            
            # Most frequent teams
            all_teams = list(home_teams.dropna()) + list(away_teams.dropna())
            team_counts = Counter(all_teams)
            print("\nMost frequent teams:")
            for team, count in team_counts.most_common(10):
                print(f"• {team}: {count} matches")
        
        # Date range
        if 'match_date' in matches_df.columns:
            print("\nDate Range:")
            dates = pd.to_datetime(matches_df['match_date'])
            print(f"From: {dates.min().strftime('%Y-%m-%d')}")
            print(f"To: {dates.max().strftime('%Y-%m-%d')}")
            print(f"Duration: {(dates.max() - dates.min()).days} days")
        
        # Match outcomes
        if 'home_score' in matches_df.columns and 'away_score' in matches_df.columns:
            print("\nMatch Outcomes:")
            home_wins = (matches_df['home_score'] > matches_df['away_score']).sum()
            away_wins = (matches_df['home_score'] < matches_df['away_score']).sum()
            draws = (matches_df['home_score'] == matches_df['away_score']).sum()
            
            total = len(matches_df)
            print(f"Home wins: {home_wins} ({home_wins/total*100:.1f}%)")
            print(f"Away wins: {away_wins} ({away_wins/total*100:.1f}%)")
            print(f"Draws: {draws} ({draws/total*100:.1f}%)")
        
        print()
    
    # Analyze events
    if 'events' in data and data['events']:
        events_data = data['events']
        print("EVENTS DATA")
        print("-" * 30)
        print(f"Matches with events: {len(events_data)}")
        
        # Count total events
        total_events = sum(len(match_events) for match_events in events_data.values())
        print(f"Total events: {total_events:,}")
        
        # Average events per match
        avg_events = total_events / len(events_data) if events_data else 0
        print(f"Average events per match: {avg_events:.0f}")
        
        # Analyze event types from a sample match
        if events_data:
            first_match_id = list(events_data.keys())[0]
            first_match_events = events_data[first_match_id]
            
            print(f"\nSample match analysis (Match ID: {first_match_id}):")
            print(f"Events in sample match: {len(first_match_events)}")
            
            if first_match_events:
                sample_event = first_match_events[0]
                print(f"Sample event keys: {list(sample_event.keys())}")
                
                # Find event type column
                type_columns = [k for k in sample_event.keys() if 'type' in k.lower()]
                if type_columns:
                    type_col = type_columns[0]
                    print(f"Event type column: '{type_col}'")
                    
                    # Count event types across all matches
                    all_event_types = []
                    for match_events in list(events_data.values())[:10]:  # Sample first 10 matches
                        for event in match_events:
                            if type_col in event:
                                event_type = event[type_col]
                                if isinstance(event_type, dict) and 'name' in event_type:
                                    all_event_types.append(event_type['name'])
                                else:
                                    all_event_types.append(str(event_type))
                    
                    if all_event_types:
                        print("\nTop event types (from sample):")
                        event_type_counts = Counter(all_event_types)
                        for event_type, count in event_type_counts.most_common(10):
                            print(f"• {event_type}: {count}")
                
                # Find time/minute columns
                time_columns = [k for k in sample_event.keys() if any(word in k.lower() for word in ['minute', 'time', 'second'])]
                if time_columns:
                    print(f"Time-related columns: {time_columns}")
                
                # Find team columns
                team_columns = [k for k in sample_event.keys() if 'team' in k.lower()]
                if team_columns:
                    print(f"Team-related columns: {team_columns}")
        
        print()
    
    # Analyze lineups if present
    if 'lineups' in data and data['lineups']:
        lineups_data = data['lineups']
        print("LINEUPS DATA")
        print("-" * 30)
        print(f"Matches with lineups: {len(lineups_data)}")
        
        # Sample lineup analysis
        if lineups_data:
            first_match_id = list(lineups_data.keys())[0]
            first_match_lineups = lineups_data[first_match_id]
            print(f"Teams in sample match: {len(first_match_lineups)}")
            
            if first_match_lineups:
                sample_lineup = first_match_lineups[0]
                print(f"Sample lineup keys: {list(sample_lineup.keys())}")
                
                if 'lineup' in sample_lineup:
                    players_count = len(sample_lineup['lineup'])
                    print(f"Players in sample team: {players_count}")
        
        print()
    
    # Data quality assessment
    print("DATA QUALITY")
    print("-" * 30)
    
    # Check data completeness
    if 'matches' in data:
        print(f"✅ Matches data: {len(data['matches'])} records")
    else:
        print("❌ No matches data found")
    
    if 'events' in data:
        matches_with_events = len(data['events'])
        total_matches = len(data.get('matches', []))
        coverage = (matches_with_events / total_matches * 100) if total_matches > 0 else 0
        print(f"✅ Events data: {matches_with_events}/{total_matches} matches ({coverage:.1f}% coverage)")
    else:
        print("❌ No events data found")
    
    if 'lineups' in data:
        matches_with_lineups = len(data['lineups'])
        total_matches = len(data.get('matches', []))
        coverage = (matches_with_lineups / total_matches * 100) if total_matches > 0 else 0
        print(f"✅ Lineups data: {matches_with_lineups}/{total_matches} matches ({coverage:.1f}% coverage)")
    else:
        print("⚠️  No lineups data found")
    
    print("\n✅ Data description complete!")

def quick_sample_analysis(json_file: str = "statsbomb_data_interim_100.json"):
    """Quick analysis of data structure"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print("QUICK DATA SAMPLE")
    print("=" * 30)
    
    # Sample match
    if 'matches' in data and data['matches']:
        print("Sample match:")
        sample_match = data['matches'][0]
        for key, value in list(sample_match.items())[:5]:
            print(f"  {key}: {value}")
        print("  ...")
    
    # Sample event
    if 'events' in data and data['events']:
        first_match_id = list(data['events'].keys())[0]
        if data['events'][first_match_id]:
            print("\nSample event:")
            sample_event = data['events'][first_match_id][0]
            for key, value in list(sample_event.items())[:5]:
                print(f"  {key}: {value}")
            print("  ...")

if __name__ == "__main__":
    # Run full description
    describe_data()
    
    #print("\n" + "="*50)
    
    # Run quick sample
    #quick_sample_analysis()
