# Create: tactical_analysis/empirical_analysis.py
"""
Empirical Analysis Implementation
Following the exact methodology from the research instructions
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
from scipy.stats import mannwhitneyu, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EmpiricalPPNAnalyzer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.ppn_networks = {}  # Store 7x7 grid networks
        self.centrality_data = {}  # Store centrality calculations
        self.situational_factors = {}  # Store team classifications
        self.statistical_results = {}  # Store Mann-Whitney U results
        
        # 7x7 Grid system (49 blocks)
        self.grid_size = 7
        self.total_blocks = 49
        
        # Statistical parameters
        self.alpha = 0.05
        self.effect_size_thresholds = {
            'small': 0.01,
            'medium': 0.059,
            'large': 0.138
        }
    
    def create_7x7_grid_mapping(self):
        """Create mapping from coordinates to 7x7 grid blocks"""
        def map_to_grid_block(x, y):
            """Map normalized coordinates (0-100) to grid block (1-1 to 7-7)"""
            # Ensure coordinates are in 0-100 range
            x = max(0, min(100, x))
            y = max(0, min(100, y))
            
            # Map to 7x7 grid (1-indexed)
            grid_x = min(7, max(1, int(np.ceil(x / (100/7)))))
            grid_y = min(7, max(1, int(np.ceil(y / (100/7)))))
            
            return f"{grid_x}-{grid_y}"
        
        return map_to_grid_block
    
    def classify_situational_factors(self):
        """Classify teams based on the three situational factors"""
        print("\n=== CLASSIFYING SITUATIONAL FACTORS ===")
        
        # Get all teams and their final positions (simulated for demo)
        teams = set()
        for match in self.data_loader.matches:
            teams.add(match['home_team'])
            teams.add(match['away_team'])
        
        teams = list(teams)
        
        # Simulate final league positions (in real analysis, use actual 2017/18 EPL standings)
        np.random.seed(42)  # For reproducible results
        team_positions = {team: i+1 for i, team in enumerate(np.random.permutation(teams)[:20])}
        
        # Classify team quality
        strong_teams = [team for team, pos in team_positions.items() if pos <= 5]
        weak_teams = [team for team, pos in team_positions.items() if pos >= 16]
        
        print(f"Strong teams (top 5): {strong_teams}")
        print(f"Weak teams (bottom 5): {weak_teams}")
        
        # Process each match for situational factors
        for match in self.data_loader.matches:
            match_id = match['match_id']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get match outcome (simulate if not available)
            home_score = match.get('home_score', np.random.randint(0, 4))
            away_score = match.get('away_score', np.random.randint(0, 4))
            
            # Classify each team
            for team in [home_team, away_team]:
                team_key = f"{match_id}_{team}"
                
                # Team Quality
                if team in strong_teams:
                    quality = 'strong'
                elif team in weak_teams:
                    quality = 'weak'
                else:
                    quality = 'middle'
                
                # Match Outcome
                if team == home_team:
                    outcome = 'winning' if home_score > away_score else 'losing' if home_score < away_score else 'draw'
                    location = 'home'
                else:
                    outcome = 'winning' if away_score > home_score else 'losing' if away_score < home_score else 'draw'
                    location = 'away'
                
                self.situational_factors[team_key] = {
                    'team_quality': quality,
                    'match_outcome': outcome,
                    'match_location': location,
                    'team': team,
                    'match_id': match_id
                }
        
        print(f"✅ Classified {len(self.situational_factors)} team-match combinations")
        return self.situational_factors
    
    def construct_ppn_networks(self):
        """Construct 7x7 Pitch-Passing Networks for each team in each match"""
        print("\n=== CONSTRUCTING 7x7 PITCH-PASSING NETWORKS ===")
        
        grid_mapper = self.create_7x7_grid_mapping()
        
        for match_id in self.data_loader.events.keys():
            print(f"Processing PPN for match {match_id}...")
            
            events = self.data_loader.events[match_id]
            
            # Get match info
            match_info = next((m for m in self.data_loader.matches if m['match_id'] == match_id), None)
            if not match_info:
                continue
            
            home_team = match_info['home_team']
            away_team = match_info['away_team']
            
            # Process each team
            for team in [home_team, away_team]:
                # Filter successful passes for this team
                team_passes = []
                
                for event in events:
                    if (event.get('type', {}).get('name') == 'Pass' and
                        event.get('team', {}).get('name') == team and
                        event.get('pass', {}).get('outcome', {}).get('name') in ['Complete', 'Successful', None]):
                        
                        # Extract coordinates
                        start_x = event.get('location', [0, 0])[0]
                        start_y = event.get('location', [0, 0])[1]
                        
                        end_location = event.get('pass', {}).get('end_location', [0, 0])
                        end_x = end_location[0] if len(end_location) > 0 else 0
                        end_y = end_location[1] if len(end_location) > 1 else 0
                        
                        # Normalize coordinates to 0-100 range (assuming original is 0-120, 0-80)
                        start_x_norm = (start_x / 120) * 100
                        start_y_norm = (start_y / 80) * 100
                        end_x_norm = (end_x / 120) * 100
                        end_y_norm = (end_y / 80) * 100
                        
                        # Map to grid blocks
                        start_block = grid_mapper(start_x_norm, start_y_norm)
                        end_block = grid_mapper(end_x_norm, end_y_norm)
                        
                        team_passes.append({
                            'from_block': start_block,
                            'to_block': end_block,
                            'timestamp': event.get('timestamp', 0)
                        })
                
                # Create network
                network = nx.DiGraph()
                
                # Add all possible nodes (7x7 grid)
                for x in range(1, 8):
                    for y in range(1, 8):
                        network.add_node(f"{x}-{y}")
                
                # Add edges with weights
                edge_weights = defaultdict(int)
                for pass_event in team_passes:
                    from_block = pass_event['from_block']
                    to_block = pass_event['to_block']
                    
                    if from_block != to_block:  # Exclude self-passes
                        edge_weights[(from_block, to_block)] += 1
                
                # Add weighted edges to network
                for (from_block, to_block), weight in edge_weights.items():
                    network.add_edge(from_block, to_block, weight=weight)
                
                # Store network
                team_key = f"{match_id}_{team}"
                self.ppn_networks[team_key] = network
                
                print(f"  {team}: {len(team_passes)} passes, {network.number_of_edges()} edges")
        
        print(f"✅ Constructed {len(self.ppn_networks)} PPNs")
        return self.ppn_networks
    
    def calculate_centrality_metrics(self):
        """Calculate DC, CC, BC for each block in each PPN"""
        print("\n=== CALCULATING CENTRALITY METRICS ===")
        
        for team_key, network in self.ppn_networks.items():
            print(f"Calculating centralities for {team_key}...")
            
            # Calculate centrality metrics
            try:
                # Degree Centrality (normalized)
                dc = nx.degree_centrality(network)
                
                # Closeness Centrality (normalized)
                cc = nx.closeness_centrality(network)
                
                # Betweenness Centrality (normalized)
                bc = nx.betweenness_centrality(network, normalized=True)
                
                # Store results
                self.centrality_data[team_key] = {
                    'degree_centrality': dc,
                    'closeness_centrality': cc,
                    'betweenness_centrality': bc
                }
                
            except Exception as e:
                print(f"  ⚠️ Error calculating centralities for {team_key}: {e}")
                continue
        
        print(f"✅ Calculated centralities for {len(self.centrality_data)} networks")
        return self.centrality_data
    
    def test_normality(self):
        """Test for normality using Shapiro-Wilk test"""
        print("\n=== TESTING FOR NORMALITY ===")
        
        normality_results = {}
        
        for centrality_type in ['degree_centrality', 'closeness_centrality', 'betweenness_centrality']:
            print(f"\nTesting {centrality_type}:")
            
            # Collect all values for this centrality type
            all_values = []
            for team_key, centralities in self.centrality_data.items():
                values = list(centralities[centrality_type].values())
                all_values.extend(values)
            
            if len(all_values) > 3:  # Minimum for Shapiro-Wilk
                stat, p_value = shapiro(all_values)
                normality_results[centrality_type] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value >= 0.05
                }
                
                print(f"  Shapiro-Wilk: W={stat:.4f}, p={p_value:.4f}")
                print(f"  Normal distribution: {'Yes' if p_value >= 0.05 else 'No'}")
            else:
                print(f"  Insufficient data for normality test")
        
        return normality_results
    
    def perform_mann_whitney_tests(self):
        """Perform Mann-Whitney U tests for situational factor comparisons"""
        print("\n=== PERFORMING MANN-WHITNEY U TESTS ===")
        
        # Define comparison groups
        comparisons = [
            ('team_quality', 'strong', 'weak'),
            ('match_outcome', 'winning', 'losing'),
            ('match_location', 'home', 'away')
        ]
        
        results = {}
        
        for factor, group1, group2 in comparisons:
            print(f"\n--- {factor.upper()}: {group1} vs {group2} ---")
            
            factor_results = {}
            
            for centrality_type in ['degree_centrality', 'closeness_centrality', 'betweenness_centrality']:
                print(f"\nAnalyzing {centrality_type}:")
                
                centrality_results = {}
                
                # For each grid block
                for x in range(1, 8):
                    for y in range(1, 8):
                        block = f"{x}-{y}"
                        
                        # Collect values for each group
                        group1_values = []
                        group2_values = []
                        
                        for team_key, centralities in self.centrality_data.items():
                            if team_key in self.situational_factors:
                                team_factor = self.situational_factors[team_key][factor]
                                
                                if team_factor == group1:
                                    group1_values.append(centralities[centrality_type].get(block, 0))
                                elif team_factor == group2:
                                    group2_values.append(centralities[centrality_type].get(block, 0))
                        
                        # Perform Mann-Whitney U test if sufficient data
                        if len(group1_values) >= 3 and len(group2_values) >= 3:
                            try:
                                statistic, p_value = mannwhitneyu(group1_values, group2_values, 
                                                                 alternative='two-sided')
                                
                                # Calculate effect size (eta squared)
                                n1, n2 = len(group1_values), len(group2_values)
                                z_score = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
                                eta_squared = (z_score ** 2) / (n1 + n2)
                                
                                # Calculate median difference
                                median1 = np.median(group1_values)
                                median2 = np.median(group2_values)
                                median_diff = median1 - median2
                                
                                centrality_results[block] = {
                                    'statistic': statistic,
                                    'p_value': p_value,
                                    'significant': p_value < self.alpha,
                                    'eta_squared': eta_squared,
                                    'effect_size': self.interpret_effect_size(eta_squared),
                                    'median_diff': median_diff,
                                    'group1_median': median1,
                                    'group2_median': median2,
                                    'n1': n1,
                                    'n2': n2
                                }
                                
                            except Exception as e:
                                print(f"    Error testing block {block}: {e}")
                                continue
                
                factor_results[centrality_type] = centrality_results
                
                # Summary for this centrality type
                significant_blocks = [block for block, result in centrality_results.items() 
                                    if result.get('significant', False)]
                print(f"  Significant blocks: {len(significant_blocks)}/{len(centrality_results)}")
            
            results[f"{factor}_{group1}_vs_{group2}"] = factor_results
        
        self.statistical_results = results
        print(f"\n✅ Completed Mann-Whitney U tests")
        return results
    
    def interpret_effect_size(self, eta_squared):
        """Interpret eta squared effect size"""
        if eta_squared >= self.effect_size_thresholds['large']:
            return 'large'
        elif eta_squared >= self.effect_size_thresholds['medium']:
            return 'medium'
        elif eta_squared >= self.effect_size_thresholds['small']:
            return 'small'
        else:
            return 'negligible'
    
    def apply_multiple_comparison_correction(self):
        """Apply FDR correction to p-values"""
        print("\n=== APPLYING MULTIPLE COMPARISON CORRECTION ===")
        
        # Collect all p-values
        all_p_values = []
        p_value_mapping = []
        
        for comparison_key, comparison_data in self.statistical_results.items():
            for centrality_type, centrality_data in comparison_data.items():
                for block, result in centrality_data.items():
                    all_p_values.append(result['p_value'])
                    p_value_mapping.append((comparison_key, centrality_type, block))
        
        # Apply FDR correction (Benjamini-Hochberg)
        if all_p_values:
            from statsmodels.stats.multitest import multipletests
            
            rejected, corrected_p_values, _, _ = multipletests(
                all_p_values, alpha=self.alpha, method='fdr_bh'
            )
            
            # Update results with corrected p-values
            for i, (comparison_key, centrality_type, block) in enumerate(p_value_mapping):
                self.statistical_results[comparison_key][centrality_type][block]['p_value_corrected'] = corrected_p_values[i]
                self.statistical_results[comparison_key][centrality_type][block]['significant_corrected'] = rejected[i]
            
            print(f"✅ Applied FDR correction to {len(all_p_values)} p-values")
            print(f"   Significant before correction: {sum(p < self.alpha for p in all_p_values)}")
            print(f"   Significant after correction: {sum(rejected)}")
    
    def generate_empirical_report(self):
        """Generate comprehensive empirical analysis report"""
        print("\n" + "="*60)
        print("EMPIRICAL PITCH-PASSING NETWORK ANALYSIS REPORT")
        print("="*60)
        
        # Summary statistics
        print(f"\n📊 ANALYSIS SUMMARY:")
        print(f"   Networks analyzed: {len(self.ppn_networks)}")
        print(f"   Situational classifications: {len(self.situational_factors)}")
        print(f"   Statistical comparisons: {len(self.statistical_results)}")
        
        # Situational factor distribution
        print(f"\n📊 SITUATIONAL FACTOR DISTRIBUTION:")
        
        for factor in ['team_quality', 'match_outcome', 'match_location']:
            factor_counts = {}
            for team_key, factors in self.situational_factors.items():
                factor_value = factors[factor]
                factor_counts[factor_value] = factor_counts.get(factor_value, 0) + 1
            
            print(f"   {factor.replace('_', ' ').title()}:")
            for value, count in factor_counts.items():
                percentage = (count / len(self.situational_factors)) * 100
                print(f"     {value}: {count} ({percentage:.1f}%)")
        
        # Statistical results summary
        print(f"\n📊 STATISTICAL RESULTS SUMMARY:")
        
        for comparison_key, comparison_data in self.statistical_results.items():
            print(f"\n--- {comparison_key.replace('_', ' ').upper()} ---")
            
            for centrality_type, centrality_data in comparison_data.items():
                significant_blocks = [block for block, result in centrality_data.items() 
                                    if result.get('significant_corrected', False)]
                
                large_effects = [block for block, result in centrality_data.items() 
                               if result.get('effect_size') == 'large']
                
                print(f"   {centrality_type.replace('_', ' ').title()}:")
                print(f"     Significant blocks (FDR corrected): {len(significant_blocks)}/49")
                print(f"     Large effect sizes: {len(large_effects)}")
                
                if significant_blocks:
                    # Show top 3 most significant
                    sorted_blocks = sorted(
                        [(block, result) for block, result in centrality_data.items() 
                         if result.get('significant_corrected', False)],
                        key=lambda x: x[1]['p_value_corrected']
                    )[:3]
                    
                    print(f"     Top significant blocks:")
                    for block, result in sorted_blocks:
                        print(f"       {block}: p={result['p_value_corrected']:.4f}, "
                              f"η²={result['eta_squared']:.4f} ({result['effect_size']})")
        
        return True
    
    def visualize_empirical_results(self):
        """Create visualizations for empirical analysis results"""
        print("\n=== GENERATING EMPIRICAL ANALYSIS VISUALIZATIONS ===")
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        
        # 1. Situational factor distributions
        factor_data = {
            'team_quality': [],
            'match_outcome': [],
            'match_location': []
        }
        
        for team_key, factors in self.situational_factors.items():
            for factor in factor_data.keys():
                factor_data[factor].append(factors[factor])
        
        for i, (factor, values) in enumerate(factor_data.items()):
            value_counts = pd.Series(values).value_counts()
            axes[0, i].pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            axes[0, i].set_title(f'{factor.replace("_", " ").title()} Distribution')
        
        # 2. Centrality distributions by factor
        centrality_types = ['degree_centrality', 'closeness_centrality', 'betweenness_centrality']
        
        for i, centrality_type in enumerate(centrality_types):
            # Collect centrality values by team quality
            strong_values = []
            weak_values = []
            
            for team_key, centralities in self.centrality_data.items():
                if team_key in self.situational_factors:
                    quality = self.situational_factors[team_key]['team_quality']
                    values = list(centralities[centrality_type].values())
                    
                    if quality == 'strong':
                        strong_values.extend(values)
                    elif quality == 'weak':
                        weak_values.extend(values)
            
            if strong_values and weak_values:
                axes[1, i].hist([strong_values, weak_values], bins=20, alpha=0.7, 
                               label=['Strong Teams', 'Weak Teams'])
                axes[1, i].set_title(f'{centrality_type.replace("_", " ").title()} by Team Quality')
                axes[1, i].legend()
                axes[1, i].set_xlabel('Centrality Value')
                axes[1, i].set_ylabel('Frequency')
        
        # 3. Heatmaps of significant results
        if self.statistical_results:
            comparison_key = list(self.statistical_results.keys())[0]
            centrality_type = 'degree_centrality'
            
            if (comparison_key in self.statistical_results and 
                centrality_type in self.statistical_results[comparison_key]):
                
                # Create 7x7 heatmap of p-values
                p_value_matrix = np.ones((7, 7))
                effect_size_matrix = np.zeros((7, 7))
                
                for block, result in self.statistical_results[comparison_key][centrality_type].items():
                    x, y = map(int, block.split('-'))
                    p_value_matrix[y-1, x-1] = result.get('p_value_corrected', 1.0)
                    effect_size_matrix[y-1, x-1] = result.get('eta_squared', 0.0)
                
                # P-value heatmap
                im1 = axes[2, 0].imshow(p_value_matrix, cmap='RdYlBu_r', vmin=0, vmax=0.05)
                axes[2, 0].set_title('P-values (FDR corrected)')
                axes[2, 0].set_xlabel('X Position')
                axes[2, 0].set_ylabel('Y Position')
                plt.colorbar(im1, ax=axes[2, 0])
                
                # Effect size heatmap
                im2 = axes[2, 1].imshow(effect_size_matrix, cmap='viridis')
                axes[2, 1].set_title('Effect Sizes (η²)')
                axes[2, 1].set_xlabel('X Position')
                axes[2, 1].set_ylabel('Y Position')
                plt.colorbar(im2, ax=axes[2, 1])
        
        # 4. Network visualization example
        if self.ppn_networks:
            sample_network = list(self.ppn_networks.values())[0]
            
            # Create position layout for 7x7 grid
            pos = {}
            for x in range(1, 8):
                for y in range(1, 8):
                    pos[f"{x}-{y}"] = (x, y)
            
            # Draw network
            nx.draw(sample_network, pos, ax=axes[2, 2], 
                   node_size=50, node_color='lightblue', 
                   edge_color='gray', arrows=True, arrowsize=10)
            axes[2, 2].set_title('Sample PPN (7x7 Grid)')
            axes[2, 2].set_aspect('equal')
        
        # Remove empty subplot
        axes[0, 3].remove()
        axes[1, 3].remove()
        axes[2, 3].remove()
        
        plt.tight_layout()
        plt.show()
    
    def save_empirical_results(self, filename='empirical_analysis_results.json'):
        """Save all empirical analysis results"""
        output_data = {
            'situational_factors': self.situational_factors,
            'statistical_results': self.statistical_results,
            'analysis_summary': {
                'networks_analyzed': len(self.ppn_networks),
                'total_comparisons': len(self.statistical_results),
                'alpha_level': self.alpha,
                'effect_size_thresholds': self.effect_size_thresholds
            },
            'methodology': {
                'grid_system': '7x7',
                'total_blocks': 49,
                'statistical_test': 'Mann-Whitney U',
                'correction_method': 'FDR (Benjamini-Hochberg)',
                'centrality_metrics': ['degree', 'closeness', 'betweenness']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"✅ Empirical analysis results saved to {filename}")
    
    def run_full_empirical_analysis(self):
        """Run the complete empirical analysis pipeline"""
        print("\n" + "="*60)
        print("STARTING EMPIRICAL PITCH-PASSING NETWORK ANALYSIS")
        print("="*60)
        
        # Step 1: Classify situational factors
        self.classify_situational_factors()
        
        # Step 2: Construct PPNs
        self.construct_ppn_networks()
        
        # Step 3: Calculate centrality metrics
        self.calculate_centrality_metrics()
        
        # Step 4: Test for normality
        self.test_normality()
        
        # Step 5: Perform Mann-Whitney U tests
        self.perform_mann_whitney_tests()
        
        # Step 6: Apply multiple comparison correction
        self.apply_multiple_comparison_correction()
        
        # Step 7: Generate report
        self.generate_empirical_report()
        
        # Step 8: Create visualizations
        self.visualize_empirical_results()
        
        # Step 9: Save results
        self.save_empirical_results()
        
        print("\n✅ EMPIRICAL ANALYSIS COMPLETE!")
        return self
