from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from .data_loader import DataLoader

class TacticalContextClassifier:
    def __init__(self, data_loader: DataLoader = None):
        self.data_loader = data_loader
        self.matches = [] if data_loader is None else data_loader.matches
        self.events = {} if data_loader is None else data_loader.events
        self.team_standings = {}
        self.context_classifications = {}
        self.context_transitions = {}

        
    
    def calculate_team_standings(self, competition_id, season_id):
        """Calculate team quality rankings based on season performance"""
        print(f"Calculating team standings for competition {competition_id}, season {season_id}...")
        
        # Get all matches for this competition/season
        comp_matches = [m for m in self.matches 
                       if m.get('competition_id') == competition_id 
                       and m.get('season_id') == season_id]
        
        team_stats = defaultdict(lambda: {'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0})
        
        for match in comp_matches:
            home_team = match['home_team']
            away_team = match['away_team']
            home_score = match['home_score']
            away_score = match['away_score']
            
            # Update stats
            team_stats[home_team]['goals_for'] += home_score
            team_stats[home_team]['goals_against'] += away_score
            team_stats[home_team]['matches'] += 1
            
            team_stats[away_team]['goals_for'] += away_score
            team_stats[away_team]['goals_against'] += home_score
            team_stats[away_team]['matches'] += 1
            
            # Calculate points (3 for win, 1 for draw, 0 for loss)
            if home_score > away_score:
                team_stats[home_team]['points'] += 3
            elif home_score < away_score:
                team_stats[away_team]['points'] += 3
            else:  # Draw
                team_stats[home_team]['points'] += 1
                team_stats[away_team]['points'] += 1
        
        # Calculate goal difference and sort teams
        for team, stats in team_stats.items():
            stats['goal_diff'] = stats['goals_for'] - stats['goals_against']
            stats['points_per_game'] = stats['points'] / max(stats['matches'], 1)
        
        # Sort teams by points, then goal difference
        sorted_teams = sorted(team_stats.items(), 
                            key=lambda x: (x[1]['points'], x[1]['goal_diff']), 
                            reverse=True)
        
        # Classify teams into quality tiers
        num_teams = len(sorted_teams)
        team_quality = {}
        
        for i, (team, stats) in enumerate(sorted_teams):
            if i < min(5, num_teams // 3):
                quality = 'Top 5'
            elif i >= num_teams - min(5, num_teams // 3):
                quality = 'Bottom 5'
            else:
                quality = 'Middle 10'
            
            team_quality[team] = {
                'ranking': i + 1,
                'quality_tier': quality,
                'points': stats['points'],
                'goal_diff': stats['goal_diff'],
                'points_per_game': stats['points_per_game']
            }
        
        comp_season_key = f"{competition_id}_{season_id}"
        self.team_standings[comp_season_key] = team_quality
        
        print(f"Team standings calculated for {len(team_quality)} teams")
        return team_quality

    def classify_score_context(self, home_score, away_score, team, is_home_team):
        """Classify score context for a team"""
        if is_home_team:
            team_score = home_score
            opponent_score = away_score
        else:
            team_score = away_score
            opponent_score = home_score
        
        score_diff = team_score - opponent_score
        
        if score_diff > 1:
            return 'Leading'
        elif score_diff < -1:
            return 'Trailing' 
        else:  # -1 <= score_diff <= 1
            return 'Tied'

    def classify_match_phase(self, minute):
        """Classify match phase based on minute"""
        if minute <= 30:
            return 'Early'
        elif minute <= 60:
            return 'Middle'
        else:
            return 'Late'

    def get_team_quality(self, team, competition_id, season_id):
        """Get team quality tier"""
        comp_season_key = f"{competition_id}_{season_id}"
        if comp_season_key in self.team_standings:
            return self.team_standings[comp_season_key].get(team, {}).get('quality_tier', 'Unknown')
        return 'Unknown'

    def calculate_match_intensity(self, match_id, window_minutes=10):
        """Calculate match intensity based on passes per minute"""
        if match_id not in self.events:
            return 'Unknown'
        
        events = self.events[match_id]
        passes = [e for e in events if e.get('type') == 'Pass']
        
        if not passes:
            return 'Low'
        
        # Get match duration
        last_minute = max([e.get('minute', 0) for e in passes])
        duration = max(last_minute, 90)  # At least 90 minutes
        
        passes_per_minute = len(passes) / duration
        
        if passes_per_minute > 15:
            return 'High'
        elif passes_per_minute >= 10:
            return 'Medium'
        else:
            return 'Low'

    def track_score_evolution(self, match_id):
        """Track how the score evolves throughout a match"""
        if match_id not in self.events:
            return None
        
        events = self.events[match_id]
        match_info = next((m for m in self.matches if m["match_id"] == match_id), None)
        if not match_info:
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        # Track goals and score evolution
        score_timeline = [(0, 0, 0)]  # (minute, home_score, away_score)
        home_score = 0
        away_score = 0
        
        for event in events:
            if event.get('type') == 'Shot' and event.get('shot_outcome') == 'Goal':
                minute = event.get('minute', 0)
                team = event.get('team', '')
                
                if team == home_team:
                    home_score += 1
                elif team == away_team:
                    away_score += 1
                
                score_timeline.append((minute, home_score, away_score))
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'score_timeline': score_timeline,
            'final_score': (home_score, away_score)
        }

    def classify_match_contexts(self, match_id):
        """Classify all contexts for a match across different time windows"""
        if match_id not in self.events:
            return None
        
        match_info = next((m for m in self.matches if m["match_id"] == match_id), None)
        if not match_info:
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        competition_id = match_info.get('competition_id')
        season_id = match_info.get('season_id')
        
        # Calculate team standings if not already done
        comp_season_key = f"{competition_id}_{season_id}"
        if comp_season_key not in self.team_standings:
            self.calculate_team_standings(competition_id, season_id)
        
        # Get score evolution
        score_evolution = self.track_score_evolution(match_id)
        match_intensity = self.calculate_match_intensity(match_id)
        
        # Time windows for classification
        time_windows = [
            ('Early', 0, 30),
            ('Middle', 30, 60), 
            ('Late', 60, 90),
            ('Full', 0, 90)
        ]
        
        match_contexts = {
            'match_info': {
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'competition':  match_info.get('competition'),
                'season': match_info.get('season'),
                'final_score': (match_info['home_score'], match_info['away_score']),
                'match_intensity': match_intensity
            },
            'contexts': {},
            'score_evolution': score_evolution
        }
        
        for window_name, start_min, end_min in time_windows:
            # Get average score during this window
            if score_evolution:
                window_scores = [s for s in score_evolution['score_timeline'] 
                               if start_min <= s[0] <= end_min]
                if window_scores:
                    avg_home_score = np.mean([s[1] for s in window_scores])
                    avg_away_score = np.mean([s[2] for s in window_scores])
                else:
                    avg_home_score = avg_away_score = 0
            else:
                # Fallback to final score proportional to window
                proportion = (end_min - start_min) / 90.0
                avg_home_score = match_info['home_score'] * proportion
                avg_away_score = match_info['away_score'] * proportion
            
            for team in [home_team, away_team]:
                is_home = team == home_team
                
                context = {
                    'time_window': (start_min, end_min),
                    'match_phase': self.classify_match_phase((start_min + end_min) / 2),
                    'score_context': self.classify_score_context(
                        avg_home_score, avg_away_score, team, is_home
                    ),
                    'team_quality': self.get_team_quality(team, competition_id, season_id),
                    'match_intensity': match_intensity,
                    'is_home_team': is_home
                }
                
                if window_name not in match_contexts['contexts']:
                    match_contexts['contexts'][window_name] = {}
                match_contexts['contexts'][window_name][team] = context
        
        self.context_classifications[match_id] = match_contexts
        return match_contexts

    def detect_context_transitions(self, match_id):
        """Detect when contexts change during a match"""
        if match_id not in self.context_classifications:
            self.classify_match_contexts(match_id)
        
        match_contexts = self.context_classifications[match_id]
        home_team = match_contexts['match_info']['home_team']
        away_team = match_contexts['match_info']['away_team']
        
        transitions = {
            home_team: {'score_transitions': [], 'phase_transitions': []},
            away_team: {'score_transitions': [], 'phase_transitions': []}
        }
        
        # Check score context transitions across phases
        phases = ['Early', 'Middle', 'Late']
        
        for team in [home_team, away_team]:
            prev_score_context = None
            prev_phase = None
            
            for phase in phases:
                if phase in match_contexts['contexts'] and team in match_contexts['contexts'][phase]:
                    current_context = match_contexts['contexts'][phase][team]
                    current_score = current_context['score_context']
                    current_phase = current_context['match_phase']
                    
                    # Score context transition
                    if prev_score_context and prev_score_context != current_score:
                        transitions[team]['score_transitions'].append({
                            'from': prev_score_context,
                            'to': current_score,
                            'phase': current_phase,
                            'minute_range': current_context['time_window']
                        })
                    
                    # Phase transition (always happens)
                    if prev_phase and prev_phase != current_phase:
                        transitions[team]['phase_transitions'].append({
                            'from': prev_phase,
                            'to': current_phase,
                            'minute_range': current_context['time_window']
                        })
                    
                    prev_score_context = current_score
                    prev_phase = current_phase
        
        self.context_transitions[match_id] = transitions
        return transitions

    def process_multiple_matches(self, match_ids=None):
        """Process context classification for multiple matches"""
        if match_ids is None:
            match_ids = list(self.events.keys())
        
        print(f"Processing context classifications for {len(match_ids)} matches...")
        
        for match_id in match_ids:
            print(f"Processing match {match_id}...")
            self.classify_match_contexts(match_id)
            self.detect_context_transitions(match_id)
        
        print("✅ Context classification complete!")
        return self.context_classifications

    def validate_context_categories(self):
        """Validate context categories with sample matches"""
        if not self.context_classifications:
            print("No context classifications to validate!")
            return
        
        print("\n=== CONTEXT VALIDATION REPORT ===")
        
        # Count context distributions
        context_stats = {
            'score_context': defaultdict(int),
            'match_phase': defaultdict(int), 
            'team_quality': defaultdict(int),
            'match_intensity': defaultdict(int)
        }
        
        transition_stats = {
            'score_transitions': defaultdict(int),
            'total_matches_with_transitions': 0
        }
        
        for match_id, match_data in self.context_classifications.items():
            # Count contexts
            for phase_name, phase_data in match_data['contexts'].items():
                if phase_name == 'Full':  # Skip full match to avoid double counting
                    continue
                    
                for team, context in phase_data.items():
                    context_stats['score_context'][context['score_context']] += 1
                    context_stats['match_phase'][context['match_phase']] += 1
                    context_stats['team_quality'][context['team_quality']] += 1
                    context_stats['match_intensity'][context['match_intensity']] += 1
            
            # Count transitions
            if match_id in self.context_transitions:
                transitions = self.context_transitions[match_id]
                has_transitions = False
                
                for team, team_transitions in transitions.items():
                    for transition in team_transitions['score_transitions']:
                        transition_key = f"{transition['from']} → {transition['to']}"
                        transition_stats['score_transitions'][transition_key] += 1
                        has_transitions = True
                
                if has_transitions:
                    transition_stats['total_matches_with_transitions'] += 1
        
        # Print validation results
        print(f"Total match-phases analyzed: {sum(context_stats['score_context'].values())}")
        
        print("\n📊 SCORE CONTEXT DISTRIBUTION:")
        for context, count in context_stats['score_context'].items():
            percentage = (count / sum(context_stats['score_context'].values())) * 100
            print(f"  {context}: {count} ({percentage:.1f}%)")
        
        print("\n📊 MATCH PHASE DISTRIBUTION:")
        for phase, count in context_stats['match_phase'].items():
            percentage = (count / sum(context_stats['match_phase'].values())) * 100
            print(f"  {phase}: {count} ({percentage:.1f}%)")
        
        print("\n📊 TEAM QUALITY DISTRIBUTION:")
        for quality, count in context_stats['team_quality'].items():
            percentage = (count / sum(context_stats['team_quality'].values())) * 100
            print(f"  {quality}: {count} ({percentage:.1f}%)")
        
        print("\n📊 MATCH INTENSITY DISTRIBUTION:")
        for intensity, count in context_stats['match_intensity'].items():
            percentage = (count / sum(context_stats['match_intensity'].values())) * 100
            print(f"  {intensity}: {count} ({percentage:.1f}%)")
        
        print(f"\n🔄 CONTEXT TRANSITIONS:")
        print(f"  Matches with score transitions: {transition_stats['total_matches_with_transitions']}")
        print(f"  Most common score transitions:")
        for transition, count in sorted(transition_stats['score_transitions'].items(), 
                                      key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {transition}: {count} times")

    def visualize_context_distributions(self):
        """Create visualizations of context distributions"""
        if not self.context_classifications:
            print("No context classifications to visualize!")
            return
        
        # Collect all context data
        contexts_data = []
        for match_id, match_data in self.context_classifications.items():
            for phase_name, phase_data in match_data['contexts'].items():
                if phase_name == 'Full':
                    continue
                for team, context in phase_data.items():
                    contexts_data.append({
                        'match_id': match_id,
                        'team': team,
                        'phase': phase_name,
                        'score_context': context['score_context'],
                        'team_quality': context['team_quality'],
                        'match_intensity': context['match_intensity'],
                        'is_home': context['is_home_team']
                    })
        
        df = pd.DataFrame(contexts_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Score context distribution
        score_counts = df['score_context'].value_counts()
        axes[0,0].pie(score_counts.values, labels=score_counts.index, autopct='%1.1f%%')
        axes[0,0].set_title('Score Context Distribution')
        
        # Team quality vs score context
        quality_score = pd.crosstab(df['team_quality'], df['score_context'])
        quality_score.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Team Quality vs Score Context')
        axes[0,1].legend(title='Score Context')
        
        # Match phase distribution  
        phase_counts = df['phase'].value_counts()
        axes[1,0].bar(phase_counts.index, phase_counts.values)
        axes[1,0].set_title('Match Phase Distribution')
        
        # Match intensity distribution
        intensity_counts = df['match_intensity'].value_counts()
        axes[1,1].bar(intensity_counts.index, intensity_counts.values)
        axes[1,1].set_title('Match Intensity Distribution')
        
        plt.tight_layout()
        plt.show()

    def save_context_data(self, filename='tactical_contexts_days3_4.json'):
        """Save context classifications and transitions"""
        output_data = {
            'context_classifications': self.context_classifications,
            'context_transitions': self.context_transitions,
            'team_standings': self.team_standings
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Context data saved to {filename}")
