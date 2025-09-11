"""
Network Motif Analysis for Tactical Pattern Recognition
"""

import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class MotifAnalyzer:
    def __init__(self, network_analyzer):
        self.network_analyzer = network_analyzer
        self.motif_patterns = {}
        self.temporal_motifs = {}
        self.context_motif_frequencies = {}
        
        # Research-backed motif types for football tactical analysis
        self.motif_types = {
            'triangle_cycle': 'Possession circulation pattern',
            'triangle_chain': 'Progressive build-up pattern', 
            'triangle_fan': 'Distribution hub pattern'
        }
    
    def identify_3node_motifs(self, network):
        """Identify 3-node motifs in network"""
        if network.number_of_nodes() < 3:
            return {}
        
        motifs = {
            'triangle_cycle': 0,    # A→B→C→A (circulation)
            'triangle_chain': 0,    # A→B→C (progressive)
            'triangle_fan': 0       # A→B, A→C (distribution)
        }
        
        # Get all 3-node combinations
        nodes = list(network.nodes())
        for node_combo in combinations(nodes, 3):
            a, b, c = node_combo
            
            # Check edge existence
            edges = {
                'ab': network.has_edge(a, b),
                'ba': network.has_edge(b, a),
                'bc': network.has_edge(b, c),
                'cb': network.has_edge(c, b),
                'ac': network.has_edge(a, c),
                'ca': network.has_edge(c, a)
            }
            
            # Classify motif type
            if edges['ab'] and edges['bc'] and edges['ca']:
                motifs['triangle_cycle'] += 1
            elif edges['ab'] and edges['bc'] and not edges['ca']:
                motifs['triangle_chain'] += 1
            elif edges['ab'] and edges['ac'] and not edges['bc']:
                motifs['triangle_fan'] += 1
        
        return motifs
    
    def extract_temporal_motifs(self, match_id, team, window_size=60):
        """Extract temporal motifs from 1-minute windows"""
        if match_id not in self.network_analyzer.context_classifier.events:
            return {}
        
        events = self.network_analyzer.context_classifier.events[match_id]
        
        # Filter team passes
        team_passes = [e for e in events 
                      if e.get('type') == 'Pass' 
                      and e.get('team') == team
                      and e.get('pass_outcome') != 'Incomplete']
        
        if len(team_passes) < 10:  # Minimum passes for analysis
            return {}
        
        # Create 1-minute windows
        max_minute = max([p.get('minute', 0) for p in team_passes])
        temporal_motifs = {}
        
        for start_minute in range(0, int(max_minute), 1):
            end_minute = start_minute + 1
            window_key = f"minute_{start_minute}_{end_minute}"
            
            # Get passes in this window
            window_passes = [p for p in team_passes 
                           if start_minute <= p.get('minute', 0) < end_minute]
            
            if len(window_passes) < 3:  # Need minimum passes for motifs
                continue
            
            # Build network for this window
            zone_passes = []
            for pass_event in window_passes:
                start_x = pass_event.get('location', [None, None])[0] if pass_event.get('location') else None
                start_y = pass_event.get('location', [None, None])[1] if pass_event.get('location') else None
                
                end_location = pass_event.get('pass_end_location', [None, None])
                end_x = end_location[0] if end_location else None
                end_y = end_location[1] if end_location else None
                
                # Map to zones using existing utility
                from .utils import map_coordinates_to_zone
                start_zone = map_coordinates_to_zone(start_x, start_y)
                end_zone = map_coordinates_to_zone(end_x, end_y)
                
                if start_zone is not None and end_zone is not None and start_zone != end_zone:
                    zone_passes.append({'from_zone': start_zone, 'to_zone': end_zone})
            
            # Build network and identify motifs
            if zone_passes:
                network = self.network_analyzer.build_zone_network(zone_passes)
                motifs = self.identify_3node_motifs(network)
                temporal_motifs[window_key] = motifs
        
        return temporal_motifs
    
    def analyze_context_specific_patterns(self, match_id):
        """Analyze how motif frequencies change under different contexts"""
        if match_id not in self.network_analyzer.context_classifier.context_classifications:
            return {}
        
        match_contexts = self.network_analyzer.context_classifier.context_classifications[match_id]
        match_info = match_contexts['match_info']
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        context_patterns = {}
        
        for team in [home_team, away_team]:
            context_patterns[team] = {}
            
            # Analyze static contexts
            for phase_name, phase_data in match_contexts['contexts'].items():
                if phase_name == 'Full' or team not in phase_data:
                    continue
                
                context = phase_data[team]
                time_window = context['time_window']
                
                # Extract motifs for this context window
                zone_passes = self.network_analyzer.extract_zone_passes(
                    match_id, team, time_window[0], time_window[1]
                )
                
                if zone_passes:
                    network = self.network_analyzer.build_zone_network(zone_passes)
                    motifs = self.identify_3node_motifs(network)
                    
                    context_key = f"{context['score_context']}_{context['match_phase']}_{context['team_quality']}"
                    context_patterns[team][context_key] = {
                        'motifs': motifs,
                        'context': context,
                        'total_passes': len(zone_passes)
                    }
        
        return context_patterns
    
    def calculate_motif_significance(self, observed_motifs, total_passes):
        """Calculate statistical significance of motif patterns"""
        if total_passes < 10:
            return {'significant': False, 'p_value': 1.0}
        
        # Expected motif frequency based on random network model
        # Research shows random networks have ~5% triangle formation rate
        expected_triangles = total_passes * 0.05
        
        total_observed = sum(observed_motifs.values())
        
        if expected_triangles == 0:
            return {'significant': False, 'p_value': 1.0}
        
        # Chi-square test for significance
        try:
            chi2_stat = (total_observed - expected_triangles) ** 2 / expected_triangles
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
            
            return {
                'significant': p_value < 0.05,
                'p_value': p_value,
                'chi2_statistic': chi2_stat,
                'observed': total_observed,
                'expected': expected_triangles
            }
        except:
            return {'significant': False, 'p_value': 1.0}
    
    def process_match_motifs(self, match_id):
        """Process complete motif analysis for a match"""
        print(f"Processing motif analysis for match {match_id}...")
        
        # Get match info
        match_info = next((m for m in self.network_analyzer.context_classifier.matches 
                          if m["match_id"] == match_id), None)
        if not match_info:
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        match_motifs = {
            'match_id': match_id,
            'teams': {home_team: {}, away_team: {}},
            'temporal_analysis': {},
            'context_analysis': {}
        }
        
        for team in [home_team, away_team]:
            # 1. Static motif analysis (using existing network data)
            if (match_id in self.network_analyzer.network_metrics and 
                team in self.network_analyzer.network_metrics[match_id]):
                
                team_networks = self.network_analyzer.zone_networks[match_id][team]
                static_motifs = {}
                
                for window_name, network in team_networks.items():
                    if window_name != 'rolling' and hasattr(network, 'nodes'):
                        motifs = self.identify_3node_motifs(network)
                        static_motifs[window_name] = motifs
                
                match_motifs['teams'][team]['static_motifs'] = static_motifs
            
            # 2. Temporal motif analysis
            temporal_motifs = self.extract_temporal_motifs(match_id, team)
            match_motifs['temporal_analysis'][team] = temporal_motifs
        
        # 3. Context-specific analysis
        context_patterns = self.analyze_context_specific_patterns(match_id)
        match_motifs['context_analysis'] = context_patterns
        
        # Store results
        self.motif_patterns[match_id] = match_motifs
        
        return match_motifs
    
    def compare_motif_contexts(self):
        """Compare motif frequencies across different contexts"""
        print("\n=== MOTIF CONTEXT COMPARISON ===")
        
        # Collect motif data by context
        context_motifs = {
            'Leading': defaultdict(list),
            'Tied': defaultdict(list), 
            'Trailing': defaultdict(list)
        }
        
        for match_id, match_data in self.motif_patterns.items():
            context_analysis = match_data.get('context_analysis', {})
            
            for team, team_contexts in context_analysis.items():
                for context_key, context_data in team_contexts.items():
                    score_context = context_data['context']['score_context']
                    motifs = context_data['motifs']
                    total_passes = context_data['total_passes']
                    
                    if total_passes > 5:  # Minimum threshold
                        for motif_type, count in motifs.items():
                            # Normalize by total passes
                            normalized_count = count / total_passes if total_passes > 0 else 0
                            context_motifs[score_context][motif_type].append(normalized_count)
        
        # Statistical comparison
        comparison_results = {}
        motif_types = ['triangle_cycle', 'triangle_chain', 'triangle_fan']
        
        for motif_type in motif_types:
            comparison_results[motif_type] = {}
            
            # Compare each pair of contexts
            contexts = ['Leading', 'Tied', 'Trailing']
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    context1, context2 = contexts[i], contexts[j]
                    
                    values1 = context_motifs[context1][motif_type]
                    values2 = context_motifs[context2][motif_type]
                    
                    if len(values1) >= 3 and len(values2) >= 3:
                        from scipy.stats import mannwhitneyu
                        statistic, p_value = mannwhitneyu(values1, values2, alternative='two-sided')
                        
                        comparison_key = f"{context1}_vs_{context2}"
                        comparison_results[motif_type][comparison_key] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'mean1': np.mean(values1),
                            'mean2': np.mean(values2),
                            'effect_size': abs(np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2)) / 2)
                        }
                        
                        if p_value < 0.05:
                            print(f"  {motif_type} - {context1} vs {context2}: p={p_value:.4f} *")
                        else:
                            print(f"  {motif_type} - {context1} vs {context2}: p={p_value:.4f}")
        
        return comparison_results
    
    def visualize_motif_patterns(self):
        """Create visualizations for motif analysis"""
        if not self.motif_patterns:
            print("No motif patterns to visualize!")
            return
        
        # Collect data for visualization
        motif_data = []
        
        for match_id, match_data in self.motif_patterns.items():
            context_analysis = match_data.get('context_analysis', {})
            
            for team, team_contexts in context_analysis.items():
                for context_key, context_data in team_contexts.items():
                    context = context_data['context']
                    motifs = context_data['motifs']
                    total_passes = context_data['total_passes']
                    
                    if total_passes > 5:
                        for motif_type, count in motifs.items():
                            motif_data.append({
                                'match_id': match_id,
                                'team': team,
                                'score_context': context['score_context'],
                                'match_phase': context['match_phase'],
                                'team_quality': context['team_quality'],
                                'motif_type': motif_type,
                                'count': count,
                                'normalized_count': count / total_passes,
                                'total_passes': total_passes
                            })
        
        if not motif_data:
            print("No data available for visualization!")
            return
        
        df = pd.DataFrame(motif_data)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Motif frequency by score context
        motif_by_context = df.groupby(['score_context', 'motif_type'])['normalized_count'].mean().unstack()
        motif_by_context.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Motif Frequency by Score Context')
        axes[0,0].set_ylabel('Normalized Frequency')
        axes[0,0].legend(title='Motif Type')
        
        # Motif patterns by match phase
        motif_by_phase = df.groupby(['match_phase', 'motif_type'])['normalized_count'].mean().unstack()
        motif_by_phase.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Motif Frequency by Match Phase')
        axes[0,1].set_ylabel('Normalized Frequency')
        axes[0,1].legend(title='Motif Type')
        
        # Team quality vs motif complexity
        sns.boxplot(data=df, x='team_quality', y='normalized_count', hue='motif_type', ax=axes[1,0])
        axes[1,0].set_title('Motif Complexity by Team Quality')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Temporal evolution (if available)
        if df['match_phase'].nunique() > 1:
            phase_order = ['Early', 'Middle', 'Late']
            phase_evolution = df.groupby(['match_phase', 'motif_type'])['normalized_count'].mean().unstack()
            phase_evolution = phase_evolution.reindex(phase_order)
            
            for motif_type in phase_evolution.columns:
                axes[1,1].plot(phase_evolution.index, phase_evolution[motif_type], 
                              marker='o', label=motif_type)
            
            axes[1,1].set_title('Motif Evolution Across Match Phases')
            axes[1,1].set_ylabel('Normalized Frequency')
            axes[1,1].legend(title='Motif Type')
        
        plt.tight_layout()
        plt.show()
    
    def process_multiple_matches(self, match_ids=None):
        """Process motif analysis for multiple matches"""
        if match_ids is None:
            match_ids = list(self.network_analyzer.network_metrics.keys())
        
        print(f"\n=== PROCESSING MOTIF ANALYSIS FOR {len(match_ids)} MATCHES ===")
        
        for i, match_id in enumerate(match_ids, 1):
            print(f"Processing match {i}/{len(match_ids)}: {match_id}")
            self.process_match_motifs(match_id)
        
        print("✅ Motif analysis complete!")
        return self.motif_patterns
    
    def save_motif_analysis(self, filename='motif_analysis_days8_9.json'):
        """Save motif analysis results"""
        import json
        
        with open(filename, 'w') as f:
            json.dump(self.motif_patterns, f, indent=2, default=str)
        
        print(f"✅ Motif analysis saved to {filename}")
