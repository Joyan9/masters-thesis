import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import create_sliding_windows, get_context_label

class ContextAnalyzer:
    """Analyzes match contexts using sliding windows"""
    
    def __init__(self, window_size=10, step_size=5, min_passes=10):
        self.window_size = window_size
        self.step_size = step_size
        self.min_passes = min_passes
        self.context_windows = {}
    
    def extract_context_windows(self, events: List[Dict], match_id: str) -> List[Dict]:
        """Extract context windows from match events"""
        df = pd.DataFrame(events)
        
        # Filter and prepare data
        df = df[df['type'].isin(['Pass', 'Shot'])].copy()
        df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
        
        # Get team names
        teams = df['team'].unique()
        if len(teams) < 2:
            print(f"Warning: Only {len(teams)} teams found in match {match_id}")
            return []
        
        # Calculate score progression
        score_progression = self._calculate_score_progression(df, teams)
        
        # Create sliding windows
        windows = create_sliding_windows(95, self.window_size, self.step_size)
        context_windows = []
        
        for start_min, end_min in windows:
            for team in teams:
                window_data = self._analyze_window(
                    df, team, start_min, end_min, score_progression, match_id
                )
                if window_data:
                    context_windows.append(window_data)
                    
        self.context_windows[match_id] = context_windows
        return context_windows
    
    def _calculate_score_progression(self, df: pd.DataFrame, teams: List[str]) -> Dict:
        """Calculate score at each minute based on goal events"""
        home_team, away_team = teams[0], teams[1]

        # Filter only shot events
        shots = df[df['type'] == 'Shot'].copy()
        
        # With flattened data structure, check the 'shot_outcome' column directly
        # Filter for goals - handle NaN values safely
        goals = shots[shots['shot_outcome'] == 'Goal'].sort_values('minute')
        
        score_progression = {}
        home_score, away_score = 0, 0

        for minute in range(91):  # 0-90
            minute_goals = goals[goals['minute'] <= minute]

            home_score = len(minute_goals[minute_goals['team'] == home_team])
            away_score = len(minute_goals[minute_goals['team'] == away_team])

            score_progression[minute] = {
                home_team: home_score,
                away_team: away_score,
                'diff': home_score - away_score
            }

        return score_progression
    
    def _analyze_window(self, df: pd.DataFrame, team: str, start_min: float, 
                       end_min: float, score_progression: Dict, match_id: str) -> Dict:
        """Analyze a single context window"""
        # Filter passes for this team and time window
        team_passes = df[
            (df['team'] == team) & 
            (df['type'] == 'Pass') &
            (df['minute'] >= start_min) & 
            (df['minute'] < end_min)
        ]
        # Check minimum pass threshold
        if len(team_passes) < self.min_passes:
            return None
        
        # Get context at window midpoint
        mid_minute = int((start_min + end_min) / 2)
        score_data = score_progression.get(mid_minute, {'diff': 0})

        # Determine score context relative to this team
        team_list = list(score_progression[0].keys())
        team_list.remove('diff')
        
        if team == team_list[0]:  # Home team
            score_diff = score_data['diff']
        else:  # Away team
            score_diff = -score_data['diff']
        
        contexts = get_context_label(mid_minute, score_diff, None)
        
        # Calculate intensity
        pass_rate = len(team_passes) / self.window_size
        
        if pass_rate > 8:
            intensity = 'high'
        elif pass_rate > 5:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        #print(f"[DEBUG] Window {start_min}-{end_min} {team}: score_diff={score_diff}, pass_rate={pass_rate:.2f}, score_context={contexts['score']}, intensity={intensity}")

        return {
            'match_id': match_id,
            'team': team,
            'start_minute': start_min,
            'end_minute': end_min,
            'pass_count': len(team_passes),
            'score_context': contexts['score'],
            'phase_context': contexts['phase'],
            'intensity_context': intensity,
            'passes': team_passes.to_dict('records')
        }
    
    def get_all_windows(self) -> List[Dict]:
        """Get all context windows from all matches"""
        all_windows = []
        for match_windows in self.context_windows.values():
            all_windows.extend(match_windows)
        return all_windows
