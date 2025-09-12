"""
Performance Impact Analysis
Days 12-14: Test whether network insights correlate with performance outcomes
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, pointbiserialr, ttest_rel, chi2_contingency
import json
from datetime import datetime

class PerformanceAnalyzer:
    def __init__(self, network_analyzer, motif_analyzer=None):
        self.network_analyzer = network_analyzer
        self.motif_analyzer = motif_analyzer
        self.performance_data = {}
        self.goal_analysis = {}
        self.correlation_results = {}
        self.performance_indicators = {}
        
        # Performance analysis parameters
        self.analysis_window = 5  # 5 minutes before/after goals
        self.zone_grid_size = 7   # 7x7 grid (49 zones)
        
        # High-impact performance indicators
        self.kpis = {
            'goal_probability': 'Network centrality → goal likelihood',
            'shot_creation_efficiency': 'Centrality patterns → xG generation', 
            'possession_retention': 'Zone-based network density → pass success',
            'defensive_vulnerability': 'Network fragmentation → goals conceded',
            'counter_attack_success': 'Transition speed metrics → scoring opportunities'
        }
    
    def extract_goal_events(self, match_id):
        """Extract open play goal events from match"""
        if match_id not in self.network_analyzer.context_classifier.events:
            return []
        
        events = self.network_analyzer.context_classifier.events[match_id]
        
        # Filter for shots that resulted in goals
        goal_events = []
        for event in events:
            if (event.get('type') == 'Shot' and 
                event.get('shot_outcome') == 'Goal' and
                event.get('play_pattern') in ['Regular Play', 'Fast Break']):  # Open play only
                
                goal_events.append({
                    'match_id': match_id,
                    'minute': event.get('minute', 0),
                    'second': event.get('second', 0),
                    'team': event.get('team'),
                    'player': event.get('player'),
                    'location': event.get('location', [None, None]),
                    'xg': event.get('shot_statsbomb_xg', 0),
                    'play_pattern': event.get('play_pattern'),
                    'timestamp': event.get('minute', 0) * 60 + event.get('second', 0)
                })
        
        return goal_events
    
    def extract_shot_events(self, match_id):
        """Extract all shot events with xG values"""
        if match_id not in self.network_analyzer.context_classifier.events:
            return []
        
        events = self.network_analyzer.context_classifier.events[match_id]
        
        shot_events = []
        for event in events:
            if event.get('type') == 'Shot' and event.get('shot_statsbomb_xg') is not None:
                shot_events.append({
                    'match_id': match_id,
                    'minute': event.get('minute', 0),
                    'second': event.get('second', 0),
                    'team': event.get('team'),
                    'player': event.get('player'),
                    'location': event.get('location', [None, None]),
                    'xg': event.get('shot_statsbomb_xg', 0),
                    'outcome': event.get('shot_outcome'),
                    'play_pattern': event.get('play_pattern'),
                    'timestamp': event.get('minute', 0) * 60 + event.get('second', 0),
                    'is_goal': event.get('shot_outcome') == 'Goal'
                })
        
        return shot_events
    
    def get_network_state_at_time(self, match_id, team, target_minute, window_size=5):
        """Get network metrics for a specific time window"""
        # Check if we have rolling window data
        if (match_id in self.network_analyzer.zone_networks and
            team in self.network_analyzer.zone_networks[match_id] and
            'rolling' in self.network_analyzer.zone_networks[match_id][team]):
            
            rolling_data = self.network_analyzer.zone_networks[match_id][team]['rolling']
            
            # Find the closest rolling window
            best_window = None
            min_distance = float('inf')
            
            for window_key, network_data in rolling_data.items():
                if 'window' in network_data:
                    window_start, window_end = network_data['window']
                    window_center = (window_start + window_end) / 2
                    distance = abs(window_center - target_minute)
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_window = window_key
            
            if best_window and min_distance <= window_size:
                return rolling_data[best_window].get('metrics', {})
        
        # Fallback: extract network for specific time window
        return self.extract_network_metrics_for_window(match_id, team, target_minute, window_size)
    
    def extract_network_metrics_for_window(self, match_id, team, center_minute, window_size):
        """Extract network metrics for a specific time window"""
        start_minute = max(0, center_minute - window_size/2)
        end_minute = center_minute + window_size/2
        
        # Extract zone passes for this window
        zone_passes = self.network_analyzer.extract_zone_passes(
            match_id, team, start_minute, end_minute
        )
        
        if not zone_passes:
            return {}
        
        # Build network and calculate metrics
        network = self.network_analyzer.build_zone_network(zone_passes)
        metrics = self.network_analyzer.calculate_network_metrics(network)
        
        return metrics
    
    def analyze_before_after_goals(self, match_id):
        """Analyze network changes before and after goals"""
        goal_events = self.extract_goal_events(match_id)
        
        if not goal_events:
            return {}
        
        match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                          if m["match_id"] == match_id), None)
        if not match_info:
            return {}
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        before_after_analysis = {
            'match_id': match_id,
            'goals': goal_events,
            'team_analysis': {home_team: [], away_team: []}
        }
        
        for goal in goal_events:
            goal_minute = goal['minute']
            scoring_team = goal['team']
            conceding_team = away_team if scoring_team == home_team else home_team
            
            # Skip goals too early or too late in the match
            if goal_minute < self.analysis_window or goal_minute > (90 - self.analysis_window):
                continue
            
            # Analyze both teams
            for team in [scoring_team, conceding_team]:
                # Get network state before goal
                before_metrics = self.get_network_state_at_time(
                    match_id, team, goal_minute - self.analysis_window, self.analysis_window
                )
                
                # Get network state after goal
                after_metrics = self.get_network_state_at_time(
                    match_id, team, goal_minute + self.analysis_window, self.analysis_window
                )
                
                if before_metrics and after_metrics:
                    goal_analysis = {
                        'goal_minute': goal_minute,
                        'goal_scorer': goal['player'],
                        'goal_xg': goal['xg'],
                        'team_role': 'scoring' if team == scoring_team else 'conceding',
                        'before_metrics': before_metrics,
                        'after_metrics': after_metrics,
                        'metric_changes': self.calculate_metric_changes(before_metrics, after_metrics)
                    }
                    
                    before_after_analysis['team_analysis'][team].append(goal_analysis)
        
        return before_after_analysis
    
    def calculate_metric_changes(self, before_metrics, after_metrics):
        """Calculate changes in network metrics"""
        changes = {}
        
        metric_keys = ['density', 'average_clustering', 'betweenness_centrality_mean', 
                      'degree_centrality_mean', 'closeness_centrality_mean', 
                      'bc_dc_ratio_mean', 'edge_weight_variance']
        
        for key in metric_keys:
            before_val = before_metrics.get(key, 0)
            after_val = after_metrics.get(key, 0)
            
            if before_val != 0:
                percent_change = ((after_val - before_val) / before_val) * 100
            else:
                percent_change = 0
            
            changes[key] = {
                'before': before_val,
                'after': after_val,
                'absolute_change': after_val - before_val,
                'percent_change': percent_change
            }
        
        return changes
    
    def analyze_shot_creation_efficiency(self, match_id):
        """Analyze relationship between network metrics and shot creation (xG)"""
        shot_events = self.extract_shot_events(match_id)
        
        if not shot_events:
            return {}
        
        match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                          if m["match_id"] == match_id), None)
        if not match_info:
            return {}
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        shot_analysis = {
            'match_id': match_id,
            'team_analysis': {home_team: [], away_team: []}
        }
        
        for team in [home_team, away_team]:
            team_shots = [s for s in shot_events if s['team'] == team]
            
            for shot in team_shots:
                shot_minute = shot['minute']
                
                # Get network state before shot (5-minute window)
                network_metrics = self.get_network_state_at_time(
                    match_id, team, shot_minute, self.analysis_window
                )
                
                if network_metrics:
                    shot_analysis['team_analysis'][team].append({
                        'shot_minute': shot_minute,
                        'shot_xg': shot['xg'],
                        'shot_outcome': shot['outcome'],
                        'is_goal': shot['is_goal'],
                        'player': shot['player'],
                        'network_metrics': network_metrics
                    })
        
        return shot_analysis
    
    def calculate_possession_retention_by_zone(self, match_id, team):
        """Calculate possession retention rates per zone"""
        if match_id not in self.network_analyzer.context_classifier.events:
            return {}
        
        events = self.network_analyzer.context_classifier.events[match_id]
        
        # Get all team passes
        team_passes = [e for e in events 
                      if e.get('type') == 'Pass' 
                      and e.get('team') == team]
        
        if not team_passes:
            return {}
        
        # Map passes to zones and track success
        from .utils import map_coordinates_to_zone
        
        zone_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        for pass_event in team_passes:
            start_location = pass_event.get('location', [None, None])
            start_x = start_location[0] if start_location else None
            start_y = start_location[1] if start_location else None
            
            start_zone = map_coordinates_to_zone(start_x, start_y)
            
            if start_zone is not None:
                zone_stats[start_zone]['total'] += 1
                
                # Check if pass was successful (no 'pass_outcome' means successful)
                if pass_event.get('pass_outcome') != 'Incomplete':
                    zone_stats[start_zone]['successful'] += 1
        
        # Calculate retention rates
        retention_rates = {}
        for zone, stats in zone_stats.items():
            if stats['total'] > 0:
                retention_rate = stats['successful'] / stats['total']
                retention_rates[zone] = {
                    'retention_rate': retention_rate,
                    'total_passes': stats['total'],
                    'successful_passes': stats['successful']
                }
        
        return retention_rates
    
    def analyze_performance_correlations(self, match_ids=None):
        """Analyze correlations between network metrics and performance outcomes"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== ANALYZING PERFORMANCE CORRELATIONS FOR {len(match_ids)} MATCHES ===")
        
        # Collect data for correlation analysis
        correlation_data = {
            'goal_probability': [],
            'shot_creation': [],
            'possession_retention': [],
            'defensive_performance': []
        }
        
        for match_id in match_ids:
            print(f"Processing correlations for match {match_id}...")
            
            # 1. Goal probability analysis
            goal_analysis = self.analyze_before_after_goals(match_id)
            if goal_analysis:
                self.goal_analysis[match_id] = goal_analysis
                
                # Extract data for correlation
                for team, team_goals in goal_analysis['team_analysis'].items():
                    for goal_data in team_goals:
                        if goal_data['team_role'] == 'scoring':
                            before_metrics = goal_data['before_metrics']
                            correlation_data['goal_probability'].append({
                                'match_id': match_id,
                                'team': team,
                                'goal_xg': goal_data['goal_xg'],
                                'bc_mean': before_metrics.get('betweenness_centrality_mean', 0),
                                'dc_mean': before_metrics.get('degree_centrality_mean', 0),
                                'cc_mean': before_metrics.get('closeness_centrality_mean', 0),
                                'density': before_metrics.get('density', 0),
                                'clustering': before_metrics.get('average_clustering', 0),
                                'bc_dc_ratio': before_metrics.get('bc_dc_ratio_mean', 0),
                                'edge_variance': before_metrics.get('edge_weight_variance', 0)
                            })
            
            # 2. Shot creation efficiency
            shot_analysis = self.analyze_shot_creation_efficiency(match_id)
            if shot_analysis:
                for team, team_shots in shot_analysis['team_analysis'].items():
                    for shot_data in team_shots:
                        metrics = shot_data['network_metrics']
                        correlation_data['shot_creation'].append({
                            'match_id': match_id,
                            'team': team,
                            'shot_xg': shot_data['shot_xg'],
                            'is_goal': shot_data['is_goal'],
                            'bc_mean': metrics.get('betweenness_centrality_mean', 0),
                            'dc_mean': metrics.get('degree_centrality_mean', 0),
                            'cc_mean': metrics.get('closeness_centrality_mean', 0),
                            'density': metrics.get('density', 0),
                            'clustering': metrics.get('average_clustering', 0),
                            'bc_dc_ratio': metrics.get('bc_dc_ratio_mean', 0),
                            'edge_variance': metrics.get('edge_weight_variance', 0)
                        })
            
            # 3. Possession retention analysis
            match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                              if m["match_id"] == match_id), None)
            if match_info:
                home_team = match_info['home_team']
                away_team = match_info['away_team']
                
                for team in [home_team, away_team]:
                    retention_data = self.calculate_possession_retention_by_zone(match_id, team)
                    
                    if retention_data:
                        # Get overall team network metrics
                        if (match_id in self.network_analyzer.network_metrics and
                            team in self.network_analyzer.network_metrics[match_id] and
                            'Full' in self.network_analyzer.network_metrics[match_id][team]):
                            
                            team_metrics = self.network_analyzer.network_metrics[match_id][team]['Full']
                            
                            # Calculate average retention rate
                            total_passes = sum([data['total_passes'] for data in retention_data.values()])
                            successful_passes = sum([data['successful_passes'] for data in retention_data.values()])
                            avg_retention = successful_passes / total_passes if total_passes > 0 else 0
                            
                            correlation_data['possession_retention'].append({
                                'match_id': match_id,
                                'team': team,
                                'retention_rate': avg_retention,
                                'total_passes': total_passes,
                                'bc_mean': team_metrics.get('betweenness_centrality_mean', 0),
                                'dc_mean': team_metrics.get('degree_centrality_mean', 0),
                                'cc_mean': team_metrics.get('closeness_centrality_mean', 0),
                                'density': team_metrics.get('density', 0),
                                'clustering': team_metrics.get('average_clustering', 0),
                                'bc_dc_ratio': team_metrics.get('bc_dc_ratio_mean', 0),
                                'edge_variance': team_metrics.get('edge_weight_variance', 0)
                            })
        
        # Calculate correlations
        self.correlation_results = self.calculate_correlation_statistics(correlation_data)
        
        return self.correlation_results
    
    def calculate_correlation_statistics(self, correlation_data):
        """Calculate correlation statistics for performance indicators"""
        results = {}
        
        # Network metric columns
        network_metrics = ['bc_mean', 'dc_mean', 'cc_mean', 'density', 'clustering', 'bc_dc_ratio', 'edge_variance']
        
        # 1. Goal probability correlations
        if correlation_data['goal_probability']:
            df_goals = pd.DataFrame(correlation_data['goal_probability'])
            goal_correlations = {}
            
            for metric in network_metrics:
                if metric in df_goals.columns and len(df_goals) > 3:
                    # Correlation with goal xG
                    corr_coef, p_value = pearsonr(df_goals[metric], df_goals['goal_xg'])
                    goal_correlations[metric] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'sample_size': len(df_goals)
                    }
            
            results['goal_probability'] = goal_correlations
        
        # 2. Shot creation correlations
        if correlation_data['shot_creation']:
            df_shots = pd.DataFrame(correlation_data['shot_creation'])
            shot_correlations = {}
            
            for metric in network_metrics:
                if metric in df_shots.columns and len(df_shots) > 3:
                    # Correlation with shot xG
                    corr_coef, p_value = pearsonr(df_shots[metric], df_shots['shot_xg'])
                    
                    # Point-biserial correlation with goal outcome
                    pb_corr, pb_p = pointbiserialr(df_shots['is_goal'], df_shots[metric])
                    
                    shot_correlations[metric] = {
                        'xg_correlation': corr_coef,
                        'xg_p_value': p_value,
                        'goal_correlation': pb_corr,
                        'goal_p_value': pb_p,
                        'xg_significant': p_value < 0.05,
                        'goal_significant': pb_p < 0.05,
                        'sample_size': len(df_shots)
                    }
            
            results['shot_creation'] = shot_correlations
        
        # 3. Possession retention correlations
        if correlation_data['possession_retention']:
            df_retention = pd.DataFrame(correlation_data['possession_retention'])
            retention_correlations = {}
            
            for metric in network_metrics:
                if metric in df_retention.columns and len(df_retention) > 3:
                    corr_coef, p_value = pearsonr(df_retention[metric], df_retention['retention_rate'])
                    retention_correlations[metric] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'sample_size': len(df_retention)
                    }
            
            results['possession_retention'] = retention_correlations
        
        return results
    
    def analyze_defensive_vulnerability(self, match_ids=None):
        """Analyze defensive vulnerability patterns"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        defensive_analysis = {}
        
        for match_id in match_ids:
            goal_events = self.extract_goal_events(match_id)
            
            if not goal_events:
                continue
            
            match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                              if m["match_id"] == match_id), None)
            if not match_info:
                continue
            
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            
            # Analyze conceding patterns
            for goal in goal_events:
                scoring_team = goal['team']
                conceding_team = away_team if scoring_team == home_team else home_team
                goal_minute = goal['minute']
                
                # Get defensive network state before goal
                defensive_metrics = self.get_network_state_at_time(
                    match_id, conceding_team, goal_minute, self.analysis_window
                )
                
                if defensive_metrics:
                    if match_id not in defensive_analysis:
                        defensive_analysis[match_id] = []
                    
                    defensive_analysis[match_id].append({
                        'conceding_team': conceding_team,
                        'goal_minute': goal_minute,
                        'goal_xg': goal['xg'],
                        'defensive_metrics': defensive_metrics,
                        'vulnerability_score': self.calculate_vulnerability_score(defensive_metrics)
                    })
        
        return defensive_analysis
    
    def calculate_vulnerability_score(self, metrics):
        """Calculate defensive vulnerability score based on network metrics"""
        # Vulnerability increases with:
        # - Low density (disconnected defense)
        # - High BC concentration (over-reliance on key players)
        # - High edge variance (inconsistent connections)
        
        density = metrics.get('density', 0.5)
        bc_concentration = metrics.get('bc_dc_ratio_mean', 0)
        edge_variance = metrics.get('edge_weight_variance', 0)
        
        # Normalize and combine (higher score = more vulnerable)
        vulnerability = (
            (1 - density) * 0.4 +  # Low density increases vulnerability
            bc_concentration * 0.3 +  # High BC concentration increases vulnerability
            edge_variance * 0.3  # High variance increases vulnerability
        )
        
        return min(1.0, vulnerability)
    
    def generate_performance_report(self, match_id):
        """Generate comprehensive performance analysis report"""
        if match_id not in self.goal_analysis:
            print(f"No goal analysis data for match {match_id}")
            return None
        
        goal_data = self.goal_analysis[match_id]
        match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                          if m["match_id"] == match_id), None)
        
        print(f"\n{'='*60}")
        print(f"PERFORMANCE ANALYSIS REPORT - MATCH {match_id}")
        print(f"{'='*60}")
        
        if match_info:
            print(f"Teams: {match_info['home_team']} vs {match_info['away_team']}")
            print(f"Final Score: {match_info['home_score']} - {match_info['away_score']}")

        
        print(f"Goals Analyzed: {len(goal_data['goals'])}")
        
        # Analyze each goal
        for i, goal in enumerate(goal_data['goals'], 1):
            print(f"\n{'-'*40}")
            print(f"GOAL {i}: {goal['player']} ({goal['team']}) - {goal['minute']}'")
            print(f"xG: {goal['xg']:.3f}")
            print(f"{'-'*40}")
            
            # Find corresponding analysis
            scoring_team = goal['team']
            if scoring_team in goal_data['team_analysis']:
                team_goals = goal_data['team_analysis'][scoring_team]
                
                # Find this specific goal
                goal_analysis = None
                for analysis in team_goals:
                    if (analysis['goal_minute'] == goal['minute'] and 
                        analysis['team_role'] == 'scoring'):
                        goal_analysis = analysis
                        break
                
                if goal_analysis:
                    print("\nNETWORK CHANGES (Before → After Goal):")
                    changes = goal_analysis['metric_changes']
                    
                    key_metrics = ['density', 'betweenness_centrality_mean', 'degree_centrality_mean']
                    for metric in key_metrics:
                        if metric in changes:
                            change_data = changes[metric]
                            print(f"  {metric.replace('_', ' ').title()}: "
                                  f"{change_data['before']:.3f} → {change_data['after']:.3f} "
                                  f"({change_data['percent_change']:+.1f}%)")
        
        # Overall correlation insights
        if self.correlation_results:
            print(f"\n{'-'*40}")
            print("PERFORMANCE CORRELATIONS")
            print(f"{'-'*40}")
            
            if 'shot_creation' in self.correlation_results:
                shot_corrs = self.correlation_results['shot_creation']
                print("\nShot Creation Efficiency:")
                
                for metric, data in shot_corrs.items():
                    if data['xg_significant']:
                        print(f"  {metric}: r={data['xg_correlation']:.3f} (p={data['xg_p_value']:.3f}) *")
                    else:
                        print(f"  {metric}: r={data['xg_correlation']:.3f} (p={data['xg_p_value']:.3f})")
        
        return goal_data
    
    def visualize_performance_analysis(self):
        """Create visualizations for performance analysis"""
        if not self.correlation_results:
            print("No correlation results to visualize!")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Goal probability correlations
        if 'goal_probability' in self.correlation_results:
            goal_corrs = self.correlation_results['goal_probability']
            metrics = list(goal_corrs.keys())
            correlations = [goal_corrs[m]['correlation'] for m in metrics]
            p_values = [goal_corrs[m]['p_value'] for m in metrics]
            
            colors = ['red' if p < 0.05 else 'blue' for p in p_values]
            
            axes[0,0].bar(range(len(metrics)), correlations, color=colors)
            axes[0,0].set_title('Goal Probability Correlations')
            axes[0,0].set_ylabel('Correlation Coefficient')
            axes[0,0].set_xticks(range(len(metrics)))
            axes[0,0].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. Shot creation correlations
        if 'shot_creation' in self.correlation_results:
            shot_corrs = self.correlation_results['shot_creation']
            metrics = list(shot_corrs.keys())
            xg_correlations = [shot_corrs[m]['xg_correlation'] for m in metrics]
            goal_correlations = [shot_corrs[m]['goal_correlation'] for m in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[0,1].bar(x - width/2, xg_correlations, width, label='xG Correlation', alpha=0.8)
            axes[0,1].bar(x + width/2, goal_correlations, width, label='Goal Correlation', alpha=0.8)
            axes[0,1].set_title('Shot Creation Correlations')
            axes[0,1].set_ylabel('Correlation Coefficient')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            axes[0,1].legend()
            axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 3. Possession retention correlations
        if 'possession_retention' in self.correlation_results:
            retention_corrs = self.correlation_results['possession_retention']
            metrics = list(retention_corrs.keys())
            correlations = [retention_corrs[m]['correlation'] for m in metrics]
            p_values = [retention_corrs[m]['p_value'] for m in metrics]
            
            colors = ['green' if p < 0.05 else 'orange' for p in p_values]
            
            axes[0,2].bar(range(len(metrics)), correlations, color=colors)
            axes[0,2].set_title('Possession Retention Correlations')
            axes[0,2].set_ylabel('Correlation Coefficient')
            axes[0,2].set_xticks(range(len(metrics)))
            axes[0,2].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
            axes[0,2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 4. Before/After goal analysis
        if self.goal_analysis:
            before_after_data = []
            
            for match_id, match_data in self.goal_analysis.items():
                for team, team_goals in match_data['team_analysis'].items():
                    for goal_data in team_goals:
                        if goal_data['team_role'] == 'scoring':
                            changes = goal_data['metric_changes']
                            before_after_data.append({
                                'bc_change': changes.get('betweenness_centrality_mean', {}).get('percent_change', 0),
                                'density_change': changes.get('density', {}).get('percent_change', 0),
                                'clustering_change': changes.get('average_clustering', {}).get('percent_change', 0)
                            })
            
            if before_after_data:
                df_changes = pd.DataFrame(before_after_data)
                
                # Box plot of metric changes
                df_changes.boxplot(ax=axes[1,0])
                axes[1,0].set_title('Network Metric Changes After Goals')
                axes[1,0].set_ylabel('Percent Change')
                axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # 5. Correlation significance heatmap
        if len(self.correlation_results) > 1:
            # Create significance matrix
            all_metrics = set()
            for analysis_type in self.correlation_results.values():
                all_metrics.update(analysis_type.keys())
            
            all_metrics = sorted(list(all_metrics))
            significance_matrix = np.zeros((len(self.correlation_results), len(all_metrics)))
            
            analysis_types = list(self.correlation_results.keys())
            
            for i, analysis_type in enumerate(analysis_types):
                for j, metric in enumerate(all_metrics):
                    if metric in self.correlation_results[analysis_type]:
                        # Use p-value for significance (lower = more significant)
                        p_val = self.correlation_results[analysis_type][metric].get('p_value', 1.0)
                        significance_matrix[i, j] = -np.log10(p_val + 1e-10)  # -log10 for better visualization
            
            im = axes[1,1].imshow(significance_matrix, cmap='Reds', aspect='auto')
            axes[1,1].set_title('Correlation Significance (-log10 p-value)')
            axes[1,1].set_xticks(range(len(all_metrics)))
            axes[1,1].set_xticklabels([m.replace('_', '\n') for m in all_metrics], rotation=45)
            axes[1,1].set_yticks(range(len(analysis_types)))
            axes[1,1].set_yticklabels([t.replace('_', ' ').title() for t in analysis_types])
            plt.colorbar(im, ax=axes[1,1])
        
        # 6. Performance summary
        axes[1,2].text(0.1, 0.9, 'PERFORMANCE SUMMARY', fontsize=14, fontweight='bold', 
                       transform=axes[1,2].transAxes)
        
        summary_text = []
        if 'goal_probability' in self.correlation_results:
            sig_count = sum(1 for data in self.correlation_results['goal_probability'].values() 
                           if data['significant'])
            summary_text.append(f"Goal Prediction: {sig_count} significant correlations")
        
        if 'shot_creation' in self.correlation_results:
            sig_count = sum(1 for data in self.correlation_results['shot_creation'].values() 
                           if data['xg_significant'])
            summary_text.append(f"Shot Creation: {sig_count} significant correlations")
        
        if 'possession_retention' in self.correlation_results:
            sig_count = sum(1 for data in self.correlation_results['possession_retention'].values() 
                           if data['significant'])
            summary_text.append(f"Possession: {sig_count} significant correlations")
        
        for i, text in enumerate(summary_text):
            axes[1,2].text(0.1, 0.7 - i*0.1, text, fontsize=10, 
                          transform=axes[1,2].transAxes)
        
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_multiple_matches(self, match_ids=None):
        """Process performance analysis for multiple matches"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== PROCESSING PERFORMANCE ANALYSIS FOR {len(match_ids)} MATCHES ===")
        
        # Analyze correlations
        self.analyze_performance_correlations(match_ids)
        
        # Analyze defensive vulnerability
        defensive_analysis = self.analyze_defensive_vulnerability(match_ids)
        self.performance_data['defensive_vulnerability'] = defensive_analysis
        
        print("✅ Performance analysis complete!")
        return self.performance_data
    
    def save_performance_analysis(self, filename='performance_analysis_days12_14.json'):
        """Save performance analysis results"""
        output_data = {
            'goal_analysis': self.goal_analysis,
            'correlation_results': self.correlation_results,
            'performance_data': self.performance_data,
            'analysis_parameters': {
                'analysis_window': self.analysis_window,
                'zone_grid_size': self.zone_grid_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Performance analysis saved to {filename}")
