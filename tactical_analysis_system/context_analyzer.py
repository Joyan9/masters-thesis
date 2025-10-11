import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import create_sliding_windows, get_context_label


class ContextAnalyzer:
    """
    Analyzes match contexts using sliding temporal windows.
    
    This class extracts context-specific time windows from match event data, enabling
    the analysis of how tactical behavior (via passing networks) varies across different
    match situations. It segments matches into overlapping time windows and labels each
    window with contextual information: score state, match phase, and passing intensity.
    
    This is a foundational component for RQ1 (Contextual Network Analysis), as it
    creates the context-labeled windows that will be used to build and compare
    passing networks.
    
    Attributes
    ----------
    window_size : int
        Duration of each time window in minutes (default: 10).
    step_size : int
        Step between consecutive window start times in minutes (default: 5).
        With window_size=10 and step_size=5, windows overlap by 50%.
    min_passes : int
        Minimum number of passes required for a window to be valid (default: 20).
        This ensures sufficient data for meaningful network construction.
        Corresponds to 2 passes/minute minimum threshold.
    context_windows : dict
        Storage for extracted windows, keyed by match_id.
    
    Notes
    -----
    **Contextual Dimensions Analyzed:**
    
    1. **Score Context**: {leading, trailing, tied}
       - Based on goal difference at window midpoint
       - Team-specific (home team's +1 = away team's -1)
    
    2. **Phase Context**: {early, middle, late}
       - Early: 0-30 minutes
       - Middle: 31-60 minutes
       - Late: 61-90+ minutes
    
    3. **Intensity Context**: {low, medium, high}
       - Based on passing rate (passes per minute)
       - Low: ≤5 passes/min
       - Medium: 5-8 passes/min
       - High: >8 passes/min
    
    **Methodological Considerations:**
    
    - **Sliding Windows**: 50% overlap ensures smooth temporal transitions and
      captures tactical changes that occur between discrete windows
    - **Minimum Pass Threshold**: 20 passes (2/min) filters out periods with
      insufficient data, based on football domain knowledge
    - **Team-Specific Analysis**: Each window is analyzed separately for each team,
      as tactical behavior is team-dependent

    """
    
    def __init__(self, window_size=10, step_size=5, min_passes=20):
        """
        Initialize ContextAnalyzer with window parameters.
        
        Parameters
        ----------
        window_size : int, default=10
            Duration of each time window in minutes. Standard value based on
            tactical analysis literature (10-minute windows capture meaningful
            tactical adjustments).
        step_size : int, default=5
            Step between consecutive windows in minutes. Default creates 50% overlap
            (step_size = window_size / 2), balancing temporal resolution with
            computational efficiency.
        min_passes : int, default=20
            Minimum passes required for valid window. Corresponds to 2 passes/minute,
            a threshold based on football domain knowledge to ensure sufficient data
            for network construction.
        
        Notes
        -----
        **Parameter Selection Rationale** (for methodology justification):
        
        - window_size=10: Captures tactical adjustments while maintaining sufficient
          sample size. Shorter windows risk insufficient data; longer windows may
          miss tactical changes.
        
        - step_size=5: 50% overlap balances temporal granularity with independence.
          Smaller steps increase correlation between windows; larger steps may miss
          transitions.
        
        - min_passes=20: Ensures network has sufficient edges for meaningful metrics.
          Based on empirical analysis showing networks with <20 passes often have
          unstable centrality measures.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.min_passes = min_passes
        self.context_windows = {}
    
    def extract_context_windows(self, events: List[Dict], match_id: str, 
                           home_team: str = None, away_team: str = None) -> List[Dict]:
        """
        Extract context-labeled windows from match events.
        
        Parameters
        ----------
        events : list of dict
                List of match event dictionaries from StatsBomb data.
        match_id : str
                Unique identifier for the match.
        home_team : str, optional
                Name of home team. If not provided, will attempt to infer from events
                (may be unreliable).
        away_team : str, optional
                Name of away team. If not provided, will attempt to infer from events
                (may be unreliable).
        
        ... rest of docstring ...
        """
        df = pd.DataFrame(events)
        
        # Filter for Pass and Shot events
        df = df[df['type'].isin(['Pass', 'Shot'])].copy()
        df['minute'] = pd.to_numeric(df['minute'], errors='coerce').fillna(0)
        
        # Get team names - use provided home/away or infer from data
        if home_team and away_team:
                teams = (home_team, away_team)
        else:
                # Fallback: infer from events (unreliable for home/away order)
                teams_in_data = df['team'].unique()
                if len(teams_in_data) < 2:
                        print(f"Warning: Only {len(teams_in_data)} teams found in match {match_id}")
                        return []
                teams = tuple(teams_in_data[:2])
                print(f"Warning: Home/away teams not provided for match {match_id}. "
                f"Inferring from data may be unreliable: {teams}")
        
        # Calculate score progression (optimized version)
        score_progression = self._calculate_score_progression(df, teams)

        # Create sliding windows (91 to ensure minute 90 is captured)
        windows = create_sliding_windows(91, self.window_size, self.step_size)
        context_windows = []
        
        # Extract windows for each team
        for start_min, end_min in windows:
            for team in teams:
                window_data = self._analyze_window(
                    df, team, start_min, end_min, score_progression, match_id, teams
                )
                if window_data:
                    context_windows.append(window_data)
                    
        self.context_windows[match_id] = context_windows
        return context_windows
    
    def _calculate_score_progression(self, df: pd.DataFrame, teams: Tuple[str, str]) -> Dict:
        """
        Calculate score at each minute based on goal events (optimized).
        
        Builds a minute-by-minute record of the match score from goal events.
        This is used to determine the score context (leading/trailing/tied) for
        each time window.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of match events, must include Shot events with 'shot_outcome'.
        teams : tuple of str
            (home_team, away_team) names. Order determined by StatsBomb data structure
            where teams[0] is home team and teams[1] is away team (per StatsBomb docs).
        
        Returns
        -------
        dict
            Dictionary mapping minute (0-90) to score information:
            {
                minute: {
                    home_team: int (goals scored),
                    away_team: int (goals scored),
                    'diff': int (home_score - away_score)
                }
            }
        
        Notes
        -----
        **Optimization**: This version is O(n log n) instead of O(n²):
        - Filters goals once
        - Sorts by minute
        - Incrementally updates scores
        
        **Team Order**: Assumes teams[0] is home, teams[1] is away based on
        StatsBomb data structure documentation. This is critical for correct
        score difference calculation.
        
        """
        home_team, away_team = teams[0], teams[1]

        # Filter for goals only
        shots = df[df['type'] == 'Shot'].copy()
        goals = shots[shots['shot_outcome'] == 'Goal'].sort_values('minute')
        
        # Initialize score progression
        score_progression = {}
        home_score, away_score = 0, 0
        
        # Convert goals to list for efficient iteration
        goal_list = goals[['minute', 'team']].values.tolist()
        goal_idx = 0
        
        # Build minute-by-minute progression (optimized O(n) approach)
        for minute in range(91):
            # Update scores for all goals that occurred up to this minute
            while goal_idx < len(goal_list) and goal_list[goal_idx][0] <= minute:
                goal_team = goal_list[goal_idx][1]
                if goal_team == home_team:
                    home_score += 1
                elif goal_team == away_team:
                    away_score += 1
                goal_idx += 1
            
            score_progression[minute] = {
                home_team: home_score,
                away_team: away_score,
                'diff': home_score - away_score
            }

        return score_progression
    
    def _analyze_window(self, df: pd.DataFrame, team: str, start_min: float, 
                       end_min: float, score_progression: Dict, match_id: str,
                       teams: Tuple[str, str]) -> Dict:
        """
        Analyze a single context window for one team.
        
        Extracts passes within the time window, calculates contextual labels,
        and packages the data for network construction.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame of match events (Pass and Shot types).
        team : str
            Team name to analyze.
        start_min : float
            Window start time in minutes.
        end_min : float
            Window end time in minutes.
        score_progression : dict
            Minute-by-minute score data from _calculate_score_progression.
        match_id : str
            Match identifier.
        teams : tuple of str
            (home_team, away_team) for consistent score difference calculation.
        
        Returns
        -------
        dict or None
            Window data dictionary if valid (≥min_passes), None otherwise.
            
            Dictionary contains:
            - 'match_id': str
            - 'team': str
            - 'start_minute': float
            - 'end_minute': float
            - 'pass_count': int
            - 'score_context': str, {leading, trailing, tied}
            - 'phase_context': str, {early, middle, late}
            - 'intensity_context': str, {low, medium, high}
            - 'passes': list of dict, pass events with location data
        
        Notes
        -----
        **Score Context Calculation**:
        - Uses window midpoint to determine score state
        - Score difference is team-specific:
          - Home team: diff = home_score - away_score
          - Away team: diff = away_score - home_score
        - This ensures 'leading' means the analyzed team is ahead
        
        **Intensity Thresholds** (passes per minute):
        - High: >8 passes/min (aggressive, possession-based play)
        - Medium: 5-8 passes/min (balanced approach)
        - Low: ≤5 passes/min (conservative, direct play)
        - Based on football analytics literature and empirical validation
        
        **Minimum Pass Filter**:
        - Windows with <min_passes are excluded (return None)
        - Ensures sufficient data for stable network metrics
        
        Examples
        --------
        >>> window = analyzer._analyze_window(
        ...     df, 'TeamA', 10, 20, score_prog, '12345', ('TeamA', 'TeamB')
        ... )
        >>> window['score_context']
        'leading'
        >>> window['intensity_context']
        'high'
        >>> len(window['passes'])
        45
        """
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
        score_data = score_progression.get(mid_minute, {teams[0]: 0, teams[1]: 0, 'diff': 0})

        # Determine score context relative to this team (fixed version)
        home_team, away_team = teams[0], teams[1]
        
        if team == home_team:
            score_diff = score_data['diff']  # Positive if home team leading
        else:  # team == away_team
            score_diff = -score_data['diff']  # Invert for away team perspective
        
        # Get score and phase contexts
        contexts = get_context_label(mid_minute, score_diff)
        
        # Calculate intensity based on passing rate
        pass_rate = len(team_passes) / self.window_size
        
        if pass_rate > 8:
            intensity = 'high'
        elif pass_rate > 5:
            intensity = 'medium'
        else:
            intensity = 'low'
        
        # Optional debug output (commented out for production)
        # print(f"[DEBUG] Window {start_min}-{end_min} {team}: score_diff={score_diff}, "
        #       f"pass_rate={pass_rate:.2f}, score_context={contexts['score']}, intensity={intensity}")

        return {
            'match_id': match_id,
            'team': team,
            'start_minute': start_min,
            'end_minute': end_min,
            'pass_count': len(team_passes),
            'score_context': contexts['score'],
            'phase_context': contexts['phase'],
            'intensity_context': intensity,
            'passes': team_passes.to_dict('records')  # Embedded for network construction
        }
    
    def get_all_windows(self) -> List[Dict]:
        """
        Retrieve all context windows from all analyzed matches.
        
        Aggregates context windows across all matches that have been processed
        by extract_context_windows(). Useful for batch analysis across multiple
        matches.
        
        Returns
        -------
        list of dict
            Combined list of all context windows from all matches.
            Each window contains the full structure from _analyze_window().
        
        Notes
        -----
        - Windows are not sorted; order depends on processing sequence
        - Each window includes 'match_id' for identification
        - Useful for creating datasets for RQ1 analysis
        
        Examples
        --------
        >>> analyzer = ContextAnalyzer()
        >>> for match_id, events in data_loader.events.items():
        ...     analyzer.extract_context_windows(events, match_id)
        >>> all_windows = analyzer.get_all_windows()
        >>> print(f"Total windows across all matches: {len(all_windows)}")
        >>> # Group by context for analysis
        >>> leading_windows = [w for w in all_windows if w['score_context'] == 'leading']
        """
        all_windows = []
        for match_windows in self.context_windows.values():
            all_windows.extend(match_windows)
        return all_windows
