import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class ContextAnalyzer:
    """Analyzes match contexts for network analysis"""
    
    def __init__(self):
        self.context_periods = {}
    
    def extract_match_contexts(self, events: List[Dict], match_id: str) -> Dict:
        """Extract contextual periods from match events"""
        df = pd.DataFrame(events)
        
        # Filter relevant events and handle missing data
        df = df[df['type'].isin(['Pass', 'Goal', 'Substitution'])].copy()
        df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
        
        contexts = {
            'score_contexts': self._get_score_contexts(df, match_id),
            'phase_contexts': self._get_phase_contexts(),
            'intensity_contexts': self._get_intensity_contexts(df)
        }
        
        self.context_periods[match_id] = contexts
        return contexts
    
    def _get_score_contexts(self, df: pd.DataFrame, match_id: str) -> List[Tuple]:
        """Determine score-based contexts (Leading/Tied/Trailing)"""
        contexts = []
        home_score, away_score = 0, 0
        last_minute = 0
        
        # Get goals chronologically
        goals = df[df['type'] == 'Goal'].sort_values('minute')
        
        # Get team names for this match
        teams = df['team'].unique()
        home_team = teams[0] if len(teams) > 0 else None
        
        for _, goal in goals.iterrows():
            minute = float(goal['minute'])
            
            # Add context period before this goal
            if minute > last_minute:
                diff = home_score - away_score
                if diff > 1:
                    state = 'leading'
                elif diff < -1:
                    state = 'trailing'
                else:
                    state = 'tied'
                contexts.append((last_minute, minute, state))
            
            # Update score (simplified - assumes first team is home team)
            if goal['team'] == home_team:
                home_score += 1
            else:
                away_score += 1
            
            last_minute = minute
        
        # Add final period
        diff = home_score - away_score
        final_state = 'leading' if diff > 1 else 'trailing' if diff < -1 else 'tied'
        contexts.append((last_minute, 90, final_state))
        
        return contexts
    
    def _get_phase_contexts(self) -> List[Tuple]:
        """Define match phases"""
        return [
            (0, 30, 'early'),
            (30, 60, 'middle'), 
            (60, 90, 'late')
        ]
    
    def _get_intensity_contexts(self, df: pd.DataFrame) -> List[Tuple]:
        """Calculate passing intensity periods"""
        passes = df[df['type'] == 'Pass']
        contexts = []
        
        for start in range(0, 90, 10):  # 10-minute windows
            end = min(start + 10, 90)
            window_passes = passes[
                (passes['minute'] >= start) & (passes['minute'] < end)
            ]
            
            pass_rate = len(window_passes) / 10  # passes per minute
            
            if pass_rate > 15:
                intensity = 'high'
            elif pass_rate > 10:
                intensity = 'medium'
            else:
                intensity = 'low'
            
            contexts.append((start, end, intensity))
        
        return contexts
