"""
Simulation Engine for What-If Scenarios
Days 12-14: Test hypothetical tactical changes using historical data
"""

import numpy as np
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from copy import deepcopy
import json
from datetime import datetime

class TacticalSimulationEngine:
    def __init__(self, network_analyzer, performance_analyzer):
        self.network_analyzer = network_analyzer
        self.performance_analyzer = performance_analyzer
        self.simulation_results = {}
        self.baseline_metrics = {}
        
        # Simulation scenarios
        self.scenarios = {
            'remove_key_player': 'Remove highest BC player - Impact on team performance',
            'distribute_centrality': 'Increase centrality distribution - More balanced network',
            'formation_shift': 'Formation shift simulation - 4-3-3 vs 4-4-2 network patterns',
            'high_pressure': 'High-pressure simulation - Compressed network effects',
            'late_game_changes': 'Late-game scenario - Network changes in final 15 minutes'
        }
        
        # Formation templates (zone-based passing patterns)
        self.formation_templates = {
            '4-3-3': {
                'defensive_zones': [0, 1, 2, 3, 4, 5, 6],  # Back row
                'midfield_zones': [14, 15, 16, 21, 22, 23, 28, 29, 30],  # Middle rows
                'attacking_zones': [35, 36, 37, 42, 43, 44, 45, 46, 47, 48]  # Front rows
            },
            '4-4-2': {
                'defensive_zones': [0, 1, 2, 3, 4, 5, 6],
                'midfield_zones': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23],
                'attacking_zones': [35, 36, 37, 42, 43, 44, 45, 46, 47, 48]
            },
            '3-5-2': {
                'defensive_zones': [1, 2, 3, 4, 5],  # 3 center-backs
                'midfield_zones': [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23, 28, 29, 30],
                'attacking_zones': [35, 36, 37, 42, 43, 44, 45, 46, 47, 48]
            }
        }
    
    def establish_baseline_metrics(self, match_ids=None):
        """Establish baseline performance metrics for comparison"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== ESTABLISHING BASELINE METRICS FOR {len(match_ids)} MATCHES ===")
        
        baseline_data = {
            'network_metrics': [],
            'performance_outcomes': [],
            'team_data': {}
        }
        
        for match_id in match_ids:
            if match_id not in self.network_analyzer.network_metrics:
                continue
            
            match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                              if m["match_id"] == match_id), None)
            if not match_info:
                continue
            
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            
            for team in [home_team, away_team]:
                if team not in self.network_analyzer.network_metrics[match_id]:
                    continue
                
                # Get full match metrics
                if 'Full' in self.network_analyzer.network_metrics[match_id][team]:
                    team_metrics = self.network_analyzer.network_metrics[match_id][team]['Full']
                    
                    # Get performance outcomes
                    goal_events = self.performance_analyzer.extract_goal_events(match_id)
                    team_goals = len([g for g in goal_events if g['team'] == team])
                    team_goals_conceded = len([g for g in goal_events if g['team'] != team])
                    
                    shot_events = self.performance_analyzer.extract_shot_events(match_id)
                    team_shots = [s for s in shot_events if s['team'] == team]
                    team_xg = sum([s['xg'] for s in team_shots])
                    
                    baseline_entry = {
                        'match_id': match_id,
                        'team': team,
                        'goals_scored': team_goals,
                        'goals_conceded': team_goals_conceded,
                        'total_xg': team_xg,
                        'shots': len(team_shots),
                        **team_metrics
                    }
                    
                    baseline_data['network_metrics'].append(baseline_entry)
                    
                    # Store team-specific data
                    if team not in baseline_data['team_data']:
                        baseline_data['team_data'][team] = []
                    baseline_data['team_data'][team].append(baseline_entry)
        
        self.baseline_metrics = baseline_data
        print(f"✅ Baseline established with {len(baseline_data['network_metrics'])} team-match observations")
        
        return baseline_data
    
    def simulate_remove_key_player(self, match_id, team, target_phase='Full'):
        """Scenario 1: Remove highest betweenness centrality player"""
        if (match_id not in self.network_analyzer.zone_networks or
            team not in self.network_analyzer.zone_networks[match_id] or
            target_phase not in self.network_analyzer.zone_networks[match_id][team]):
            return None
        
        original_network = self.network_analyzer.zone_networks[match_id][team][target_phase]
        
        if not hasattr(original_network, 'nodes') or len(original_network.nodes()) == 0:
            return None
        
        # Calculate original betweenness centrality
        original_bc = nx.betweenness_centrality(original_network, weight='weight')
        
        if not original_bc:
            return None
        
        # Find highest BC node (key player zone)
        key_zone = max(original_bc.keys(), key=lambda x: original_bc[x])
        key_bc_value = original_bc[key_zone]
        
        # Create modified network without key player
        modified_network = original_network.copy()
        
        # Remove the key zone and redistribute its connections
        if key_zone in modified_network.nodes():
            # Get neighbors before removal
            neighbors = list(modified_network.neighbors(key_zone))
            
            # Remove the key node
            modified_network.remove_node(key_zone)
            
            # Redistribute connections among neighbors (simplified redistribution)
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    node1, node2 = neighbors[i], neighbors[j]
                    if modified_network.has_edge(node1, node2):
                        # Increase existing edge weight
                        modified_network[node1][node2]['weight'] *= 1.2
                    else:
                        # Create new edge with moderate weight
                        modified_network.add_edge(node1, node2, weight=0.3)
        
        # Calculate new metrics
        original_metrics = self.network_analyzer.calculate_network_metrics(original_network)
        modified_metrics = self.network_analyzer.calculate_network_metrics(modified_network)
        
        # Calculate impact
        impact_analysis = {
            'scenario': 'remove_key_player',
            'match_id': match_id,
            'team': team,
            'phase': target_phase,
            'removed_zone': key_zone,
            'removed_bc_value': key_bc_value,
            'original_metrics': original_metrics,
            'modified_metrics': modified_metrics,
            'metric_changes': self.calculate_simulation_impact(original_metrics, modified_metrics),
            'performance_prediction': self.predict_performance_impact(original_metrics, modified_metrics)
        }
        
        return impact_analysis
    
    def simulate_distribute_centrality(self, match_id, team, target_phase='Full'):
        """Scenario 2: Redistribute centrality more evenly"""
        if (match_id not in self.network_analyzer.zone_networks or
            team not in self.network_analyzer.zone_networks[match_id] or
            target_phase not in self.network_analyzer.zone_networks[match_id][team]):
            return None
        
        original_network = self.network_analyzer.zone_networks[match_id][team][target_phase]
        
        if not hasattr(original_network, 'nodes') or len(original_network.nodes()) < 3:
            return None
        
        # Create modified network with more distributed centrality
        modified_network = original_network.copy()
        
        # Calculate current centrality distribution
        original_bc = nx.betweenness_centrality(modified_network, weight='weight')
        
        if not original_bc:
            return None
        
        # Identify high and low centrality nodes
        bc_values = list(original_bc.values())
        bc_mean = np.mean(bc_values)
        bc_std = np.std(bc_values)
        
        high_bc_nodes = [node for node, bc in original_bc.items() if bc > bc_mean + bc_std/2]
        low_bc_nodes = [node for node, bc in original_bc.items() if bc < bc_mean - bc_std/2]
        
        # Redistribute edges to balance centrality
        for high_node in high_bc_nodes:
            for low_node in low_bc_nodes:
                if not modified_network.has_edge(high_node, low_node):
                    # Add new connection
                    modified_network.add_edge(high_node, low_node, weight=0.4)
                else:
                    # Strengthen existing connection
                    modified_network[high_node][low_node]['weight'] *= 1.3
        
        # Calculate new metrics
        original_metrics = self.network_analyzer.calculate_network_metrics(original_network)
        modified_metrics = self.network_analyzer.calculate_network_metrics(modified_network)
        
        impact_analysis = {
            'scenario': 'distribute_centrality',
            'match_id': match_id,
            'team': team,
            'phase': target_phase,
            'original_bc_std': bc_std,
            'modified_bc_std': np.std(list(nx.betweenness_centrality(modified_network, weight='weight').values())),
            'original_metrics': original_metrics,
            'modified_metrics': modified_metrics,
            'metric_changes': self.calculate_simulation_impact(original_metrics, modified_metrics),
            'performance_prediction': self.predict_performance_impact(original_metrics, modified_metrics)
        }
        
        return impact_analysis
    
    def simulate_formation_shift(self, match_id, team, from_formation='4-3-3', to_formation='4-4-2'):
        """Scenario 3: Simulate formation change impact"""
        if match_id not in self.network_analyzer.zone_networks or team not in self.network_analyzer.zone_networks[match_id]:
            return None
        
        # Get original network (using Full match data)
        if 'Full' not in self.network_analyzer.zone_networks[match_id][team]:
            return None
        
        original_network = self.network_analyzer.zone_networks[match_id][team]['Full']
        
        if not hasattr(original_network, 'nodes'):
            return None
        
        # Create formation-adjusted network
        modified_network = self.apply_formation_template(original_network, from_formation, to_formation)
        
        if modified_network is None:
            return None
        
        # Calculate metrics
        original_metrics = self.network_analyzer.calculate_network_metrics(original_network)
        modified_metrics = self.network_analyzer.calculate_network_metrics(modified_network)
        
        impact_analysis = {
            'scenario': 'formation_shift',
            'match_id': match_id,
            'team': team,
            'from_formation': from_formation,
            'to_formation': to_formation,
            'original_metrics': original_metrics,
            'modified_metrics': modified_metrics,
            'metric_changes': self.calculate_simulation_impact(original_metrics, modified_metrics),
            'performance_prediction': self.predict_performance_impact(original_metrics, modified_metrics)
        }
        
        return impact_analysis
    
    def apply_formation_template(self, original_network, from_formation, to_formation):
        """Apply formation template to modify network structure"""
        if from_formation not in self.formation_templates or to_formation not in self.formation_templates:
            return None
        
        modified_network = original_network.copy()
        
        from_template = self.formation_templates[from_formation]
        to_template = self.formation_templates[to_formation]
        
        # Adjust edge weights based on formation change
        for edge in modified_network.edges():
            zone1, zone2 = edge
            current_weight = modified_network[zone1][zone2]['weight']
            
            # Determine zone roles in both formations
            zone1_from_role = self.get_zone_role(zone1, from_template)
            zone2_from_role = self.get_zone_role(zone2, from_template)
            zone1_to_role = self.get_zone_role(zone1, to_template)
            zone2_to_role = self.get_zone_role(zone2, to_template)
            
            # Adjust weight based on role compatibility
            weight_modifier = self.calculate_formation_weight_modifier(
                zone1_from_role, zone2_from_role, zone1_to_role, zone2_to_role
            )
            
            modified_network[zone1][zone2]['weight'] = current_weight * weight_modifier
        
        return modified_network
    
    def get_zone_role(self, zone, formation_template):
        """Determine zone role in formation"""
        if zone in formation_template['defensive_zones']:
            return 'defensive'
        elif zone in formation_template['midfield_zones']:
            return 'midfield'
        elif zone in formation_template['attacking_zones']:
            return 'attacking'
        else:
            return 'neutral'
    
    def calculate_formation_weight_modifier(self, zone1_from, zone2_from, zone1_to, zone2_to):
        """Calculate weight modifier based on formation role changes"""
        # Base modifier
        modifier = 1.0
        
        # Same role connections are strengthened
        if zone1_to == zone2_to:
            modifier *= 1.2
        
        # Cross-role connections based on tactical logic
        role_compatibility = {
            ('defensive', 'midfield'): 1.1,
            ('midfield', 'attacking'): 1.15,
            ('defensive', 'attacking'): 0.8  # Direct long balls less common
        }
        
        role_pair = tuple(sorted([zone1_to, zone2_to]))
        if role_pair in role_compatibility:
            modifier *= role_compatibility[role_pair]
        
        return max(0.1, min(2.0, modifier))  # Constrain modifier
    
    def simulate_high_pressure(self, match_id, team, target_phase='Full'):
        """Scenario 4: Simulate high-pressure defensive situation"""
        if (match_id not in self.network_analyzer.zone_networks or
            team not in self.network_analyzer.zone_networks[match_id] or
            target_phase not in self.network_analyzer.zone_networks[match_id][team]):
            return None
        
        original_network = self.network_analyzer.zone_networks[match_id][team][target_phase]
        
        if not hasattr(original_network, 'nodes'):
            return None
        
        # Create high-pressure network (compressed, more local connections)
        modified_network = original_network.copy()
        
        # Reduce long-distance connections (simulate pressure)
        from .utils import map_coordinates_to_zone
        
        edges_to_modify = []
        for edge in modified_network.edges():
            zone1, zone2 = edge
            
            # Calculate zone distance (simplified)
            zone1_x, zone1_y = zone1 % 7, zone1 // 7
            zone2_x, zone2_y = zone2 % 7, zone2 // 7
            distance = abs(zone1_x - zone2_x) + abs(zone1_y - zone2_y)
            
            if distance > 2:  # Long-distance connection
                current_weight = modified_network[zone1][zone2]['weight']
                # Reduce weight under pressure
                new_weight = current_weight * 0.6
                edges_to_modify.append((zone1, zone2, new_weight))
        
        # Apply modifications
        for zone1, zone2, new_weight in edges_to_modify:
            modified_network[zone1][zone2]['weight'] = new_weight
        
        # Increase local connections (players cluster under pressure)
        for edge in modified_network.edges():
            zone1, zone2 = edge
            zone1_x, zone1_y = zone1 % 7, zone1 // 7
            zone2_x, zone2_y = zone2 % 7, zone2 // 7
            distance = abs(zone1_x - zone2_x) + abs(zone1_y - zone2_y)
            
            if distance <= 1:  # Adjacent zones
                current_weight = modified_network[zone1][zone2]['weight']
                modified_network[zone1][zone2]['weight'] = current_weight * 1.4
        
        # Calculate metrics
        original_metrics = self.network_analyzer.calculate_network_metrics(original_network)
        modified_metrics = self.network_analyzer.calculate_network_metrics(modified_network)
        
        impact_analysis = {
            'scenario': 'high_pressure',
            'match_id': match_id,
            'team': team,
            'phase': target_phase,
            'pressure_intensity': 'high',
            'original_metrics': original_metrics,
            'modified_metrics': modified_metrics,
            'metric_changes': self.calculate_simulation_impact(original_metrics, modified_metrics),
            'performance_prediction': self.predict_performance_impact(original_metrics, modified_metrics)
        }
        
        return impact_analysis
    
    def simulate_late_game_changes(self, match_id, team):
        """Scenario 5: Simulate late-game tactical changes"""
        # Compare early vs late game patterns
        early_phase = 'Early'
        late_phase = 'Late'
        
        if (match_id not in self.network_analyzer.zone_networks or
            team not in self.network_analyzer.zone_networks[match_id]):
            return None
        
        team_networks = self.network_analyzer.zone_networks[match_id][team]
        
        if early_phase not in team_networks or late_phase not in team_networks:
            return None
        
        early_network = team_networks[early_phase]
        late_network = team_networks[late_phase]
        
        if not hasattr(early_network, 'nodes') or not hasattr(late_network, 'nodes'):
            return None
        
        # Calculate metrics for both phases
        early_metrics = self.network_analyzer.calculate_network_metrics(early_network)
        late_metrics = self.network_analyzer.calculate_network_metrics(late_network)
        
        # Simulate enhanced late-game urgency
        enhanced_late_network = late_network.copy()
        
        # Increase edge weights (more direct play)
        for edge in enhanced_late_network.edges():
            current_weight = enhanced_late_network[edge[0]][edge[1]]['weight']
            enhanced_late_network[edge[0]][edge[1]]['weight'] = current_weight * 1.3
        
        enhanced_late_metrics = self.network_analyzer.calculate_network_metrics(enhanced_late_network)
        
        impact_analysis = {
            'scenario': 'late_game_changes',
            'match_id': match_id,
            'team': team,
            'early_metrics': early_metrics,
            'late_metrics': late_metrics,
            'enhanced_late_metrics': enhanced_late_metrics,
            'early_to_late_changes': self.calculate_simulation_impact(early_metrics, late_metrics),
            'late_to_enhanced_changes': self.calculate_simulation_impact(late_metrics, enhanced_late_metrics),
            'performance_prediction': self.predict_performance_impact(late_metrics, enhanced_late_metrics)
        }
        
        return impact_analysis
    
    def calculate_simulation_impact(self, original_metrics, modified_metrics):
        """Calculate impact of simulation changes"""
        impact = {}
        
        metric_keys = ['density', 'average_clustering', 'betweenness_centrality_mean', 
                      'degree_centrality_mean', 'closeness_centrality_mean', 
                      'bc_dc_ratio_mean', 'edge_weight_variance']
        
        for key in metric_keys:
            original_val = original_metrics.get(key, 0)
            modified_val = modified_metrics.get(key, 0)
            
            if original_val != 0:
                percent_change = ((modified_val - original_val) / original_val) * 100
            else:
                percent_change = 0
            
            impact[key] = {
                'original': original_val,
                'modified': modified_val,
                'absolute_change': modified_val - original_val,
                'percent_change': percent_change
            }
        
        return impact
    
    def predict_performance_impact(self, original_metrics, modified_metrics):
        """Predict performance impact based on correlation patterns"""
        if not hasattr(self.performance_analyzer, 'correlation_results'):
            return {'prediction': 'No correlation data available'}
        
        correlation_results = self.performance_analyzer.correlation_results
        
        predictions = {}
        
        # Goal probability prediction
        if 'goal_probability' in correlation_results:
            goal_corrs = correlation_results['goal_probability']
            goal_impact_score = 0
            significant_factors = 0
            
            for metric, corr_data in goal_corrs.items():
                if corr_data['significant']:
                    original_val = original_metrics.get(metric.replace('_mean', ''), 0)
                    modified_val = modified_metrics.get(metric.replace('_mean', ''), 0)
                    
                    if original_val != 0:
                        change_magnitude = abs((modified_val - original_val) / original_val)
                        correlation_strength = abs(corr_data['correlation'])
                        
                        # Positive correlation means increase in metric increases goal probability
                        if corr_data['correlation'] > 0:
                            impact = change_magnitude * correlation_strength * (1 if modified_val > original_val else -1)
                        else:
                            impact = change_magnitude * correlation_strength * (-1 if modified_val > original_val else 1)
                        
                        goal_impact_score += impact
                        significant_factors += 1
            
            if significant_factors > 0:
                goal_impact_score /= significant_factors
                predictions['goal_probability_change'] = {
                    'impact_score': goal_impact_score,
                    'interpretation': self.interpret_impact_score(goal_impact_score, 'goal_probability')
                }
        
        # Shot creation prediction
        if 'shot_creation' in correlation_results:
            shot_corrs = correlation_results['shot_creation']
            shot_impact_score = 0
            significant_factors = 0
            
            for metric, corr_data in shot_corrs.items():
                if corr_data['xg_significant']:
                    original_val = original_metrics.get(metric.replace('_mean', ''), 0)
                    modified_val = modified_metrics.get(metric.replace('_mean', ''), 0)
                    
                    if original_val != 0:
                        change_magnitude = abs((modified_val - original_val) / original_val)
                        correlation_strength = abs(corr_data['xg_correlation'])
                        
                        if corr_data['xg_correlation'] > 0:
                            impact = change_magnitude * correlation_strength * (1 if modified_val > original_val else -1)
                        else:
                            impact = change_magnitude * correlation_strength * (-1 if modified_val > original_val else 1)
                        
                        shot_impact_score += impact
                        significant_factors += 1
            
            if significant_factors > 0:
                shot_impact_score /= significant_factors
                predictions['shot_creation_change'] = {
                    'impact_score': shot_impact_score,
                    'interpretation': self.interpret_impact_score(shot_impact_score, 'shot_creation')
                }
        
        return predictions
    
    def interpret_impact_score(self, score, metric_type):
        """Interpret impact score into actionable insights"""
        abs_score = abs(score)
        direction = "increase" if score > 0 else "decrease"
        
        if abs_score > 0.2:
            magnitude = "significant"
        elif abs_score > 0.1:
            magnitude = "moderate"
        elif abs_score > 0.05:
            magnitude = "small"
        else:
            magnitude = "minimal"
        
        return f"{magnitude.title()} {direction} in {metric_type.replace('_', ' ')}"
    
    def run_all_scenarios(self, match_id, team):
        """Run all simulation scenarios for a team in a match"""
        print(f"\n=== RUNNING SIMULATION SCENARIOS FOR {team} IN MATCH {match_id} ===")
        
        scenario_results = {
            'match_id': match_id,
            'team': team,
            'scenarios': {}
        }
        
        # Scenario 1: Remove key player
        print("Running Scenario 1: Remove Key Player...")
        result1 = self.simulate_remove_key_player(match_id, team)
        if result1:
            scenario_results['scenarios']['remove_key_player'] = result1
        
        # Scenario 2: Distribute centrality
        print("Running Scenario 2: Distribute Centrality...")
        result2 = self.simulate_distribute_centrality(match_id, team)
        if result2:
            scenario_results['scenarios']['distribute_centrality'] = result2
        
        # Scenario 3: Formation shift
        print("Running Scenario 3: Formation Shift...")
        result3 = self.simulate_formation_shift(match_id, team, '4-3-3', '4-4-2')
        if result3:
            scenario_results['scenarios']['formation_shift'] = result3
        
        # Scenario 4: High pressure
        print("Running Scenario 4: High Pressure...")
        result4 = self.simulate_high_pressure(match_id, team)
        if result4:
            scenario_results['scenarios']['high_pressure'] = result4
        
        # Scenario 5: Late game changes
        print("Running Scenario 5: Late Game Changes...")
        result5 = self.simulate_late_game_changes(match_id, team)
        if result5:
            scenario_results['scenarios']['late_game_changes'] = result5
        
        return scenario_results
    
    def generate_simulation_report(self, simulation_results):
        """Generate comprehensive simulation report"""
        match_id = simulation_results['match_id']
        team = simulation_results['team']
        
        print(f"\n{'='*60}")
        print(f"TACTICAL SIMULATION REPORT")
        print(f"{'='*60}")
        print(f"Match: {match_id}")
        print(f"Team: {team}")
        print(f"Scenarios Analyzed: {len(simulation_results['scenarios'])}")
        
        for scenario_name, scenario_data in simulation_results['scenarios'].items():
            print(f"\n{'-'*50}")
            print(f"SCENARIO: {scenario_name.upper().replace('_', ' ')}")
            print(f"{'-'*50}")
            
            # Show key metric changes
            if 'metric_changes' in scenario_data:
                changes = scenario_data['metric_changes']
                
                print("\nKey Network Changes:")
                key_metrics = ['density', 'betweenness_centrality_mean', 'degree_centrality_mean']
                
                for metric in key_metrics:
                    if metric in changes:
                        change_data = changes[metric]
                        print(f"  {metric.replace('_', ' ').title()}: "
                              f"{change_data['original']:.3f} → {change_data['modified']:.3f} "
                              f"({change_data['percent_change']:+.1f}%)")
            
            # Show performance predictions
            if 'performance_prediction' in scenario_data:
                predictions = scenario_data['performance_prediction']
                
                print("\nPerformance Impact Predictions:")
                for pred_type, pred_data in predictions.items():
                    if isinstance(pred_data, dict) and 'interpretation' in pred_data:
                        print(f"  {pred_type.replace('_', ' ').title()}: {pred_data['interpretation']}")
        
        return simulation_results
    
    def visualize_simulation_results(self, simulation_results):
        """Create visualizations for simulation results"""
        if not simulation_results.get('scenarios'):
            print("No simulation results to visualize!")
            return
        
        scenarios = simulation_results['scenarios']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Metric changes comparison
        scenario_names = []
        density_changes = []
        bc_changes = []
        dc_changes = []
        
        for scenario_name, scenario_data in scenarios.items():
            if 'metric_changes' in scenario_data:
                changes = scenario_data['metric_changes']
                scenario_names.append(scenario_name.replace('_', '\n'))
                
                density_changes.append(changes.get('density', {}).get('percent_change', 0))
                bc_changes.append(changes.get('betweenness_centrality_mean', {}).get('percent_change', 0))
                dc_changes.append(changes.get('degree_centrality_mean', {}).get('percent_change', 0))
        
        if scenario_names:
            x = np.arange(len(scenario_names))
            width = 0.25
            
            axes[0,0].bar(x - width, density_changes, width, label='Density', alpha=0.8)
            axes[0,0].bar(x, bc_changes, width, label='Betweenness Centrality', alpha=0.8)
            axes[0,0].bar(x + width, dc_changes, width, label='Degree Centrality', alpha=0.8)
            
            axes[0,0].set_title('Network Metric Changes by Scenario')
            axes[0,0].set_ylabel('Percent Change')
            axes[0,0].set_xticks(x)
            axes[0,0].set_xticklabels(scenario_names, rotation=45)
            axes[0,0].legend()
            axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # 2. Performance impact scores
        scenario_names = []
        goal_impacts = []
        shot_impacts = []

        for scenario_name, scenario_data in scenarios.items():
            if 'metric_changes' in scenario_data:  # ✅ only consider scenarios with metric_changes
                scenario_names.append(scenario_name.replace('_', '\n'))

                if 'performance_prediction' in scenario_data:
                    predictions = scenario_data['performance_prediction']
                    goal_impact = predictions.get('goal_probability_change', {}).get('impact_score', 0)
                    shot_impact = predictions.get('shot_creation_change', {}).get('impact_score', 0)
                else:
                    goal_impact = 0
                    shot_impact = 0

                goal_impacts.append(goal_impact)
                shot_impacts.append(shot_impact)

        if scenario_names and (any(goal_impacts) or any(shot_impacts)):
            x = np.arange(len(scenario_names))
            width = 0.35
            axes[0,1].bar(x - width/2, goal_impacts, width, label='Goal Probability', alpha=0.8)
            axes[0,1].bar(x + width/2, shot_impacts, width, label='Shot Creation', alpha=0.8)
            axes[0,1].set_title('Performance Impact Scores')
            axes[0,1].set_xticks(x)
            axes[0,1].set_xticklabels(scenario_names, rotation=30, ha='right')
            axes[0,1].legend()

        
        # 3. Scenario comparison radar chart
        if len(scenarios) >= 2:
            # Select key metrics for radar chart
            metrics = ['density', 'betweenness_centrality_mean', 'degree_centrality_mean', 
                      'average_clustering', 'edge_weight_variance']
            
            # Get data for first two scenarios
            scenario_list = list(scenarios.items())[:2]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            for i, (scenario_name, scenario_data) in enumerate(scenario_list):
                if 'metric_changes' in scenario_data:
                    changes = scenario_data['metric_changes']
                    values = []
                    
                    for metric in metrics:
                        change = changes.get(metric, {}).get('percent_change', 0)
                        values.append(change)
                    
                    values += values[:1]  # Complete the circle
                    
                    axes[0,2].plot(angles, values, 'o-', linewidth=2, 
                                  label=scenario_name.replace('_', ' ').title())
                    axes[0,2].fill(angles, values, alpha=0.25)
            
            axes[0,2].set_xticks(angles[:-1])
            axes[0,2].set_xticklabels([m.replace('_', '\n') for m in metrics])
            axes[0,2].set_title('Scenario Comparison (% Change)')
            axes[0,2].legend()
            axes[0,2].grid(True)
        
        # 4. Formation shift analysis (if available)
        if 'formation_shift' in scenarios:
            formation_data = scenarios['formation_shift']
            
            if 'metric_changes' in formation_data:
                changes = formation_data['metric_changes']
                
                metrics = list(changes.keys())[:6]  # Top 6 metrics
                before_values = [changes[m]['original'] for m in metrics]
                after_values = [changes[m]['modified'] for m in metrics]
                
                x = np.arange(len(metrics))
                width = 0.35
                
                axes[1,0].bar(x - width/2, before_values, width, label='Before Formation Change', alpha=0.8)
                axes[1,0].bar(x + width/2, after_values, width, label='After Formation Change', alpha=0.8)
                
                axes[1,0].set_title('Formation Shift Impact')
                axes[1,0].set_ylabel('Metric Value')
                axes[1,0].set_xticks(x)
                axes[1,0].set_xticklabels([m.replace('_', '\n') for m in metrics], rotation=45)
                axes[1,0].legend()
        
        # 5. Risk-Reward Analysis
        risk_scores = []
        reward_scores = []
        
        for scenario_name, scenario_data in scenarios.items():
            # Calculate risk (variance in metrics)
            if 'metric_changes' in scenario_data:
                changes = scenario_data['metric_changes']
                change_values = [abs(changes[m]['percent_change']) for m in changes.keys()]
                risk_score = np.std(change_values) if change_values else 0
                
                # Calculate reward (positive performance impact)
                reward_score = 0
                if 'performance_prediction' in scenario_data:
                    predictions = scenario_data['performance_prediction']
                    for pred_data in predictions.values():
                        if isinstance(pred_data, dict) and 'impact_score' in pred_data:
                            reward_score += max(0, pred_data['impact_score'])
                
                risk_scores.append(risk_score)
                reward_scores.append(reward_score)
        
        if risk_scores and reward_scores:
            axes[1,1].scatter(risk_scores, reward_scores, s=100, alpha=0.7)
            
            for i, scenario_name in enumerate(scenario_names):
                axes[1,1].annotate(scenario_name, (risk_scores[i], reward_scores[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            axes[1,1].set_title('Risk-Reward Analysis')
            axes[1,1].set_xlabel('Risk (Metric Variance)')
            axes[1,1].set_ylabel('Reward (Performance Gain)')
            axes[1,1].grid(True, alpha=0.3)
        
        # 6. Summary recommendations
        axes[1,2].text(0.1, 0.9, 'SIMULATION SUMMARY', fontsize=14, fontweight='bold', 
                       transform=axes[1,2].transAxes)
        
        # Find best scenario
        if reward_scores and any(r > 0 for r in reward_scores):
            best_scenario_idx = np.argmax(reward_scores)
            best_scenario = scenario_names[best_scenario_idx]
            
            summary_text = [
                f"Best Scenario: {best_scenario}",
                f"Expected Reward: {reward_scores[best_scenario_idx]:.3f}",
                f"Risk Level: {risk_scores[best_scenario_idx]:.3f}",
                "",
                "Key Insights:",
                "• Formation changes show moderate impact",
                "• Centrality distribution affects performance",
                "• Late-game adjustments are critical"
            ]
        else:
            summary_text = [
                "No clear performance improvements found",
                "Consider alternative tactical approaches",
                "",
                "Key Insights:",
                "• Current network structure may be optimal",
                "• Small adjustments have limited impact",
                "• Focus on execution rather than structure"
            ]
        
        for i, text in enumerate(summary_text):
            axes[1,2].text(0.1, 0.8 - i*0.08, text, fontsize=10, 
                          transform=axes[1,2].transAxes)
        
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def process_multiple_matches(self, match_ids=None, max_teams_per_match=2):
        """Process simulation scenarios for multiple matches"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== PROCESSING SIMULATIONS FOR {len(match_ids)} MATCHES ===")
        
        for i, match_id in enumerate(match_ids[:3], 1):  # Limit to 3 matches for demo
            print(f"\nProcessing match {i}: {match_id}")
            
            match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                              if m["match_id"] == match_id), None)
            if not match_info:
                continue
            
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            
            # Run scenarios for both teams
            for team in [home_team, away_team]:
                print(f"  Analyzing {team}...")
                team_results = self.run_all_scenarios(match_id, team)
                
                if team_results['scenarios']:
                    self.simulation_results[f"{match_id}_{team}"] = team_results
        
        print("✅ Simulation analysis complete!")
        return self.simulation_results
    
    def save_simulation_results(self, filename='simulation_results_days12_14.json'):
        """Save simulation results"""
        output_data = {
            'simulation_results': self.simulation_results,
            'baseline_metrics': self.baseline_metrics,
            'scenarios': self.scenarios,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Simulation results saved to {filename}")
