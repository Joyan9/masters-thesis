import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import mannwhitneyu
from collections import defaultdict
from .utils import map_coordinates_to_zone, create_rolling_windows

class BaselineNetworkAnalyzer:
    def __init__(self, context_classifier):
        self.context_classifier = context_classifier
        self.zone_networks = {}  # Store networks by match_id, team, and time_window
        self.centrality_data = {}  # Store centrality measures
        self.vulnerability_signatures = {}  # Store vulnerability metrics
        self.network_metrics = {}  # Store all network metrics
        
        # Research-backed thresholds
        self.vulnerability_thresholds = {
            'edge_weight_change': 30,  # 30% change indicates significant tactical shift
            'bc_dc_ratio_change': 25,  # 25% change in BC/DC ratio indicates reorganization
            'density_change': 15,      # 15% change in network density
            'centrality_shift': 20     # 20% change in centrality measures
        }
    
    def extract_zone_passes(self, match_id, team, start_minute, end_minute):
        """Extract passes between 7x7 zones for given time window and team"""
        if match_id not in self.context_classifier.events:
            return []
        
        events = self.context_classifier.events[match_id]
        zone_passes = []
        
        for event in events:
            # Filter for successful passes by the specified team in time window
            if (event.get('type') == 'Pass' and 
                event.get('team') == team and
                event.get('pass_outcome') != 'Incomplete' and  # Successful passes only
                start_minute <= event.get('minute', 0) <= end_minute):
                
                # Get pass coordinates
                start_x = event.get('location', [None, None])[0] if event.get('location') else None
                start_y = event.get('location', [None, None])[1] if event.get('location') else None
                
                end_location = event.get('pass_end_location', [None, None])
                end_x = end_location[0] if end_location else None
                end_y = end_location[1] if end_location else None
                
                # Map to zones
                start_zone = self.map_coordinates_to_zone(start_x, start_y)
                end_zone = self.map_coordinates_to_zone(end_x, end_y)
                
                if start_zone is not None and end_zone is not None and start_zone != end_zone:
                    zone_passes.append({
                        'from_zone': start_zone,
                        'to_zone': end_zone,
                        'minute': event.get('minute', 0),
                        'player': event.get('player', 'Unknown')
                    })
        
        return zone_passes
    
    def build_zone_network(self, zone_passes):
        """Build weighted network from zone passes"""
        G = nx.DiGraph()
        
        # Add all possible zones as nodes (0-48 for 7x7 grid)
        G.add_nodes_from(range(49))
        
        # Count passes between zones
        edge_weights = defaultdict(int)
        for pass_data in zone_passes:
            edge = (pass_data['from_zone'], pass_data['to_zone'])
            edge_weights[edge] += 1
        
        # Add edges with weights
        for (from_zone, to_zone), weight in edge_weights.items():
            G.add_edge(from_zone, to_zone, weight=weight)
        
        return G
    
    def calculate_network_metrics(self, network):
        """Calculate centrality measures and network properties"""
        if network.number_of_nodes() == 0:
            return None
        
        metrics = {}
        
        try:
            # Centrality measures
            metrics['betweenness_centrality'] = nx.betweenness_centrality(network, weight='weight')
            metrics['degree_centrality'] = nx.degree_centrality(network)
            metrics['closeness_centrality'] = nx.closeness_centrality(network)
            #metrics['eigenvector_centrality'] = nx.eigenvector_centrality(network, weight='weight', max_iter=1000)
            try:
                metrics['eigenvector_centrality'] = nx.eigenvector_centrality(
                    network, weight='weight', max_iter=5000, tol=1e-06
                )
            except nx.PowerIterationFailedConvergence:
                # Fallback: use numpy solver or assign zeros
                try:
                    metrics['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(
                        network, weight='weight'
                    )
                except Exception:
                    metrics['eigenvector_centrality'] = {n: 0 for n in network.nodes()}

            # Network properties
            metrics['density'] = nx.density(network)
            metrics['number_of_edges'] = network.number_of_edges()
            metrics['average_clustering'] = nx.average_clustering(network, weight='weight')
            
            # Edge weight statistics
            edge_weights = [data['weight'] for _, _, data in network.edges(data=True)]
            if edge_weights:
                metrics['edge_weight_mean'] = np.mean(edge_weights)
                metrics['edge_weight_std'] = np.std(edge_weights)
                metrics['edge_weight_variance'] = np.var(edge_weights)
            else:
                metrics['edge_weight_mean'] = 0
                metrics['edge_weight_std'] = 0
                metrics['edge_weight_variance'] = 0
            
            # BC/DC ratio for each node
            bc_values = list(metrics['betweenness_centrality'].values())
            dc_values = list(metrics['degree_centrality'].values())
            
            bc_dc_ratios = []
            for bc, dc in zip(bc_values, dc_values):
                if dc > 0:
                    bc_dc_ratios.append(bc / dc)
                else:
                    bc_dc_ratios.append(0)
            
            metrics['bc_dc_ratio_mean'] = np.mean(bc_dc_ratios) if bc_dc_ratios else 0
            metrics['bc_dc_ratio_std'] = np.std(bc_dc_ratios) if bc_dc_ratios else 0
            
        except Exception as e:
            print(f"Error calculating network metrics: {e}")
            return None
        
        return metrics
    
    def analyze_match_networks(self, match_id):
        """Analyze networks for a single match across different time windows"""
        if match_id not in self.context_classifier.events:
            print(f"No events data for match {match_id}")
            return None
        
        match_info = next((m for m in self.context_classifier.matches if m["match_id"] == match_id), None)
        if not match_info:
            print(f"No match info for match {match_id}")
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        # Initialize storage for this match
        self.zone_networks[match_id] = {}
        self.network_metrics[match_id] = {}
        
        # Static analysis windows (from context classifier)
        static_windows = [
            ('Early', 0, 30),
            ('Middle', 30, 60),
            ('Late', 60, 90),
            ('Full', 0, 90)
        ]
        
        # Rolling windows for dynamic analysis
        rolling_windows = self.create_rolling_windows()
        
        for team in [home_team, away_team]:
            self.zone_networks[match_id][team] = {}
            self.network_metrics[match_id][team] = {}
            
            # Static analysis
            for window_name, start_min, end_min in static_windows:
                zone_passes = self.extract_zone_passes(match_id, team, start_min, end_min)
                network = self.build_zone_network(zone_passes)
                metrics = self.calculate_network_metrics(network)
                
                self.zone_networks[match_id][team][window_name] = network
                self.network_metrics[match_id][team][window_name] = metrics
            
            # Dynamic analysis (rolling windows)
            self.zone_networks[match_id][team]['rolling'] = {}
            self.network_metrics[match_id][team]['rolling'] = {}
            
            for i, (start_min, end_min) in enumerate(rolling_windows):
                window_key = f"window_{i}_{start_min}_{end_min}"
                zone_passes = self.extract_zone_passes(match_id, team, start_min, end_min)
                network = self.build_zone_network(zone_passes)
                metrics = self.calculate_network_metrics(network)
                
                self.zone_networks[match_id][team]['rolling'][window_key] = network
                self.network_metrics[match_id][team]['rolling'][window_key] = metrics
        
        print(f"✅ Network analysis complete for match {match_id}")
        return self.network_metrics[match_id]
    
    def detect_vulnerability_signatures(self, match_id):
        """Identify tactical vulnerability signatures for a match"""
        if match_id not in self.network_metrics:
            print(f"No network metrics for match {match_id}. Running analysis...")
            self.analyze_match_networks(match_id)
        
        match_info = next((m for m in self.context_classifier.matches if m["match_id"] == match_id), None)
        if not match_info:
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        vulnerabilities = {
            home_team: {'signatures': [], 'severity': 'Low'},
            away_team: {'signatures': [], 'severity': 'Low'}
        }
        
        for team in [home_team, away_team]:
            if team not in self.network_metrics[match_id]:
                continue
            
            team_metrics = self.network_metrics[match_id][team]
            
            # Analyze static windows for sudden changes
            static_phases = ['Early', 'Middle', 'Late']
            for i in range(len(static_phases) - 1):
                current_phase = static_phases[i]
                next_phase = static_phases[i + 1]
                
                if (current_phase in team_metrics and next_phase in team_metrics and
                    team_metrics[current_phase] and team_metrics[next_phase]):
                    
                    current = team_metrics[current_phase]
                    next_metrics = team_metrics[next_phase]
                    
                    # Check edge weight variance change
                    if current['edge_weight_variance'] > 0:
                        variance_change = abs(next_metrics['edge_weight_variance'] - current['edge_weight_variance']) / current['edge_weight_variance'] * 100
                        if variance_change > self.vulnerability_thresholds['edge_weight_change']:
                            vulnerabilities[team]['signatures'].append({
                                'type': 'Edge Weight Variance Spike',
                                'phase_transition': f"{current_phase} → {next_phase}",
                                'change_percentage': variance_change,
                                'severity': 'High' if variance_change > 50 else 'Medium'
                            })
                    
                    # Check BC/DC ratio change
                    if current['bc_dc_ratio_mean'] > 0:
                        ratio_change = abs(next_metrics['bc_dc_ratio_mean'] - current['bc_dc_ratio_mean']) / current['bc_dc_ratio_mean'] * 100
                        if ratio_change > self.vulnerability_thresholds['bc_dc_ratio_change']:
                            vulnerabilities[team]['signatures'].append({
                                'type': 'BC/DC Ratio Shift',
                                'phase_transition': f"{current_phase} → {next_phase}",
                                'change_percentage': ratio_change,
                                'severity': 'High' if ratio_change > 40 else 'Medium'
                            })
                    
                    # Check network density change
                    if current['density'] > 0:
                        density_change = abs(next_metrics['density'] - current['density']) / current['density'] * 100
                        if density_change > self.vulnerability_thresholds['density_change']:
                            vulnerabilities[team]['signatures'].append({
                                'type': 'Network Density Change',
                                'phase_transition': f"{current_phase} → {next_phase}",
                                'change_percentage': density_change,
                                'severity': 'Medium' if density_change > 25 else 'Low'
                            })
            
            # Analyze rolling windows for dynamic patterns
            if 'rolling' in team_metrics:
                rolling_metrics = team_metrics['rolling']
                rolling_keys = sorted([k for k in rolling_metrics.keys() if rolling_metrics[k] is not None])
                
                for i in range(len(rolling_keys) - 1):
                    current_key = rolling_keys[i]
                    next_key = rolling_keys[i + 1]
                    
                    current = rolling_metrics[current_key]
                    next_metrics = rolling_metrics[next_key]
                    
                    if current and next_metrics:
                        # Check for sudden centrality shifts
                        bc_change = abs(next_metrics['bc_dc_ratio_mean'] - current['bc_dc_ratio_mean'])
                        if current['bc_dc_ratio_mean'] > 0:
                            bc_change_pct = bc_change / current['bc_dc_ratio_mean'] * 100
                            if bc_change_pct > self.vulnerability_thresholds['centrality_shift']:
                                vulnerabilities[team]['signatures'].append({
                                    'type': 'Dynamic Centrality Shift',
                                    'window_transition': f"{current_key} → {next_key}",
                                    'change_percentage': bc_change_pct,
                                    'severity': 'High' if bc_change_pct > 35 else 'Medium'
                                })
            
            # Determine overall severity
            high_severity = len([s for s in vulnerabilities[team]['signatures'] if s['severity'] == 'High'])
            medium_severity = len([s for s in vulnerabilities[team]['signatures'] if s['severity'] == 'Medium'])
            
            if high_severity >= 2:
                vulnerabilities[team]['severity'] = 'Critical'
            elif high_severity >= 1 or medium_severity >= 3:
                vulnerabilities[team]['severity'] = 'High'
            elif medium_severity >= 1:
                vulnerabilities[team]['severity'] = 'Medium'
            else:
                vulnerabilities[team]['severity'] = 'Low'
        
        self.vulnerability_signatures[match_id] = vulnerabilities
        return vulnerabilities
    
    def compare_contexts_static(self):
        """Static comparison of network metrics across contexts using Mann-Whitney U tests"""
        print("\n=== STATIC CONTEXT COMPARISON ===")
        
        # Collect data by context
        context_data = {
            'score_context': defaultdict(list),
            'match_phase': defaultdict(list),
            'team_quality': defaultdict(list),
            'match_intensity': defaultdict(list)
        }
        
        # Collect centrality data for each context
        for match_id in self.network_metrics:
            if match_id not in self.context_classifier.context_classifications:
                continue
            
            match_contexts = self.context_classifier.context_classifications[match_id]
            
            for phase_name, phase_data in match_contexts['contexts'].items():
                if phase_name == 'Full':
                    continue
                
                for team, context in phase_data.items():
                    if (team in self.network_metrics[match_id] and 
                        phase_name in self.network_metrics[match_id][team] and
                        self.network_metrics[match_id][team][phase_name] is not None):
                        
                        metrics = self.network_metrics[match_id][team][phase_name]
                        
                        # Store metrics by context
                        context_data['score_context'][context['score_context']].append(metrics)
                        context_data['match_phase'][context['match_phase']].append(metrics)
                        context_data['team_quality'][context['team_quality']].append(metrics)
                        context_data['match_intensity'][context['match_intensity']].append(metrics)
        
        # Perform statistical tests
        statistical_results = {}
        
        for context_type, context_groups in context_data.items():
            statistical_results[context_type] = {}
            
            # Get all unique context values
            contexts = list(context_groups.keys())
            
            if len(contexts) < 2:
                continue
            
            print(f"\n📊 {context_type.upper()} COMPARISON:")
            
            # Compare each pair of contexts
            for i in range(len(contexts)):
                for j in range(i + 1, len(contexts)):
                    context1, context2 = contexts[i], contexts[j]
                    group1_metrics = context_groups[context1]
                    group2_metrics = context_groups[context2]
                    
                    if len(group1_metrics) < 3 or len(group2_metrics) < 3:
                        continue
                    
                    comparison_key = f"{context1}_vs_{context2}"
                    statistical_results[context_type][comparison_key] = {}
                    
                    # Test different network metrics
                    metrics_to_test = ['density', 'bc_dc_ratio_mean', 'edge_weight_variance', 'average_clustering']
                    
                    for metric in metrics_to_test:
                        group1_values = [m[metric] for m in group1_metrics if metric in m and m[metric] is not None]
                        group2_values = [m[metric] for m in group2_metrics if metric in m and m[metric] is not None]
                        
                        if len(group1_values) >= 3 and len(group2_values) >= 3:
                            # Mann-Whitney U test
                            statistic, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                            
                            # Effect size (η²)
                            n1, n2 = len(group1_values), len(group2_values)
                            eta_squared = (statistic - (n1 * n2) / 2) ** 2 / ((n1 * n2) * (n1 + n2 + 1) / 12)
                            
                            statistical_results[context_type][comparison_key][metric] = {
                                'statistic': statistic,
                                'p_value': p_value,
                                'eta_squared': eta_squared,
                                'group1_mean': np.mean(group1_values),
                                'group2_mean': np.mean(group2_values),
                                'significant': p_value < 0.05
                            }
                            
                            if p_value < 0.05:
                                print(f"  {context1} vs {context2} ({metric}): p={p_value:.4f}, η²={eta_squared:.4f} *")
                            else:
                                print(f"  {context1} vs {context2} ({metric}): p={p_value:.4f}, η²={eta_squared:.4f}")
        
        return statistical_results
    
    def analyze_dynamic_patterns(self, match_id):
        """Analyze dynamic patterns in 10-minute rolling windows"""
        if match_id not in self.network_metrics:
            print(f"No network metrics for match {match_id}")
            return None
        
        match_info = next((m for m in self.context_classifier.matches if m["match_id"] == match_id), None)
        if not match_info:
            return None
        
        home_team = match_info['home_team']
        away_team = match_info['away_team']
        
        dynamic_patterns = {
            home_team: {'trends': [], 'volatility': {}},
            away_team: {'trends': [], 'volatility': {}}
        }
        
        for team in [home_team, away_team]:
            if (team not in self.network_metrics[match_id] or 
                'rolling' not in self.network_metrics[match_id][team]):
                continue
            
            rolling_metrics = self.network_metrics[match_id][team]['rolling']
            rolling_keys = sorted([k for k in rolling_metrics.keys() if rolling_metrics[k] is not None])
            
            if len(rolling_keys) < 3:
                continue
            
            # Extract time series for key metrics
            time_series = {
                'density': [],
                'bc_dc_ratio_mean': [],
                'edge_weight_variance': [],
                'average_clustering': []
            }
            
            for key in rolling_keys:
                metrics = rolling_metrics[key]
                for metric_name in time_series.keys():
                    if metric_name in metrics and metrics[metric_name] is not None:
                        time_series[metric_name].append(metrics[metric_name])
                    else:
                        time_series[metric_name].append(0)
            
            # Analyze trends and volatility
            for metric_name, values in time_series.items():
                if len(values) >= 3:
                    # Calculate trend (slope)
                    x = np.arange(len(values))
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                    
                    # Calculate volatility (coefficient of variation)
                    volatility = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
                    
                    dynamic_patterns[team]['trends'].append({
                        'metric': metric_name,
                        'slope': slope,
                        'r_squared': r_value ** 2,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'trend_strength': abs(slope)
                    })
                    
                    dynamic_patterns[team]['volatility'][metric_name] = {
                        'coefficient_of_variation': volatility,
                        'standard_deviation': np.std(values),
                        'mean': np.mean(values),
                        'volatility_level': 'High' if volatility > 0.3 else 'Medium' if volatility > 0.15 else 'Low'
                    }
        
        return dynamic_patterns
    
    def process_multiple_matches(self, match_ids=None):
        """Process network analysis for multiple matches"""
        if match_ids is None:
            match_ids = list(self.context_classifier.events.keys())
        
        print(f"\n=== PROCESSING NETWORK ANALYSIS FOR {len(match_ids)} MATCHES ===")
        
        for i, match_id in enumerate(match_ids, 1):
            print(f"Processing match {i}/{len(match_ids)}: {match_id}")
            self.analyze_match_networks(match_id)
            self.detect_vulnerability_signatures(match_id)
        
        print("✅ Network analysis complete for all matches!")
        return self.network_metrics
    
    def generate_vulnerability_report(self):
        """Generate comprehensive vulnerability analysis report"""
        print("\n=== TACTICAL VULNERABILITY ANALYSIS REPORT ===")
        
        if not self.vulnerability_signatures:
            print("No vulnerability signatures detected!")
            return
        
        # Summary statistics
        total_matches = len(self.vulnerability_signatures)
        total_teams = total_matches * 2
        
        vulnerability_counts = {
            'Critical': 0,
            'High': 0,
            'Medium': 0,
            'Low': 0
        }
        
        signature_types = defaultdict(int)
        
        for match_id, match_vulnerabilities in self.vulnerability_signatures.items():
            for team, team_data in match_vulnerabilities.items():
                vulnerability_counts[team_data['severity']] += 1
                
                for signature in team_data['signatures']:
                    signature_types[signature['type']] += 1
        
        print(f"📊 VULNERABILITY SEVERITY DISTRIBUTION ({total_teams} team-matches):")
        for severity, count in vulnerability_counts.items():
            percentage = (count / total_teams) * 100
            print(f"  {severity}: {count} ({percentage:.1f}%)")
        
        print(f"\n📊 MOST COMMON VULNERABILITY SIGNATURES:")
        for sig_type, count in sorted(signature_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {sig_type}: {count} occurrences")
        
        # Sample detailed analysis
        print(f"\n📋 SAMPLE DETAILED ANALYSIS:")
        sample_match = list(self.vulnerability_signatures.keys())[0]
        sample_vulnerabilities = self.vulnerability_signatures[sample_match]
        
        print(f"Match {sample_match}:")
        for team, team_data in sample_vulnerabilities.items():
            print(f"\n  {team} (Severity: {team_data['severity']}):")
            for signature in team_data['signatures'][:3]:  # Show first 3 signatures
                print(f"    - {signature['type']}: {signature['change_percentage']:.1f}% change")
    
    def visualize_network_analysis(self):
        """Create visualizations for network analysis results"""
        if not self.network_metrics:
            print("No network metrics to visualize!")
            return
        
        # Collect data for visualization
        all_metrics = []
        for match_id in self.network_metrics:
            if match_id not in self.context_classifier.context_classifications:
                continue
            
            match_contexts = self.context_classifier.context_classifications[match_id]
            
            for phase_name, phase_data in match_contexts['contexts'].items():
                if phase_name == 'Full':
                    continue
                
                for team, context in phase_data.items():
                    if (team in self.network_metrics[match_id] and 
                        phase_name in self.network_metrics[match_id][team] and
                        self.network_metrics[match_id][team][phase_name] is not None):
                        
                        metrics = self.network_metrics[match_id][team][phase_name]
                        
                        all_metrics.append({
                            'match_id': match_id,
                            'team': team,
                            'phase': phase_name,
                            'score_context': context['score_context'],
                            'team_quality': context['team_quality'],
                            'density': metrics.get('density', 0),
                            'bc_dc_ratio': metrics.get('bc_dc_ratio_mean', 0),
                            'edge_weight_variance': metrics.get('edge_weight_variance', 0),
                            'clustering': metrics.get('average_clustering', 0)
                        })
        
        if not all_metrics:
            print("No data available for visualization!")
            return
        
        df = pd.DataFrame(all_metrics)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Network density by score context
        sns.boxplot(data=df, x='score_context', y='density', ax=axes[0,0])
        axes[0,0].set_title('Network Density by Score Context')
        
        # BC/DC ratio by team quality
        sns.boxplot(data=df, x='team_quality', y='bc_dc_ratio', ax=axes[0,1])
        axes[0,1].set_title('BC/DC Ratio by Team Quality')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Edge weight variance by match phase
        sns.boxplot(data=df, x='phase', y='edge_weight_variance', ax=axes[0,2])
        axes[0,2].set_title('Edge Weight Variance by Match Phase')
        
        # Clustering coefficient by score context
        sns.boxplot(data=df, x='score_context', y='clustering', ax=axes[1,0])
        axes[1,0].set_title('Clustering Coefficient by Score Context')
        
        # Correlation heatmap
        corr_data = df[['density', 'bc_dc_ratio', 'edge_weight_variance', 'clustering']].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,1])
        axes[1,1].set_title('Network Metrics Correlation')
        
        # Vulnerability severity distribution
        if self.vulnerability_signatures:
            severity_data = []
            for match_vulnerabilities in self.vulnerability_signatures.values():
                for team_data in match_vulnerabilities.values():
                    severity_data.append(team_data['severity'])
            
            severity_counts = pd.Series(severity_data).value_counts()
            axes[1,2].pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%')
            axes[1,2].set_title('Vulnerability Severity Distribution')
        
        plt.tight_layout()
        plt.show()
    
    def save_network_analysis(self, filename='network_analysis_days5_7.json'):
        """Save network analysis results"""
        # Convert NetworkX graphs to serializable format
        serializable_networks = {}
        for match_id, match_data in self.zone_networks.items():
            serializable_networks[match_id] = {}
            for team, team_data in match_data.items():
                serializable_networks[match_id][team] = {}
                for window, network in team_data.items():
                    if isinstance(network, dict):
                        # Handle rolling windows
                        serializable_networks[match_id][team][window] = {}
                        for sub_window, sub_network in network.items():
                            if hasattr(sub_network, 'edges'):
                                serializable_networks[match_id][team][window][sub_window] = {
                                    'nodes': list(sub_network.nodes()),
                                    'edges': [(u, v, d) for u, v, d in sub_network.edges(data=True)]
                                }
                    else:
                        # Handle regular networks
                        if hasattr(network, 'edges'):
                            serializable_networks[match_id][team][window] = {
                                'nodes': list(network.nodes()),
                                'edges': [(u, v, d) for u, v, d in network.edges(data=True)]
                            }
        
        output_data = {
            'network_metrics': self.network_metrics,
            'vulnerability_signatures': self.vulnerability_signatures,
            'vulnerability_thresholds': self.vulnerability_thresholds,
            'zone_networks': serializable_networks
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Network analysis data saved to {filename}")
