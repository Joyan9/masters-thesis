import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List

class NetworkAnalyzer:
    """Calculates network metrics for contextual analysis"""
    
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.results = {}
    
    def analyze_contextual_networks(self, networks: Dict, match_id: str) -> pd.DataFrame:
        """Calculate network metrics for all contexts"""
        results = []
        
        for context_type, context_networks in networks.items():
            for context_label, network_data in context_networks.items():
                network = network_data['network']
                period = network_data['period']
                pass_count = network_data['pass_count']
                
                metrics = self._calculate_network_metrics(network)
                
                result = {
                    'match_id': match_id,
                    'context_type': context_type,
                    'context_label': context_label,
                    'start_minute': period[0],
                    'end_minute': period[1],
                    'pass_count': pass_count,
                    **metrics
                }
                results.append(result)
        
        df_results = pd.DataFrame(results)
        self.results[match_id] = df_results
        return df_results
    
    def _calculate_network_metrics(self, G: nx.Graph) -> Dict:
        """Calculate core network metrics"""
        if G.number_of_edges() == 0:
            return {
                'density': 0,
                'clustering_coefficient': 0,
                'avg_betweenness_centrality': 0,
                'avg_eigenvector_centrality': 0,
                'avg_path_length': 0,
                'centralization': 0,
                'normalized_path_length': 0,
                'normalized_centralization': 0
            }
        
        # Basic metrics
        density = nx.density(G)
        clustering = nx.average_clustering(G, weight='weight')
        
        # Centrality measures
        betweenness = nx.betweenness_centrality(G, weight='weight')
        
        try:
            eigenvector = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except:
            eigenvector = {node: 0 for node in G.nodes()}
        
        # Average centralities (only for nodes with edges)
        active_nodes = [n for n in G.nodes() if G.degree(n) > 0]
        
        avg_betweenness = np.mean([betweenness[n] for n in active_nodes]) if active_nodes else 0
        avg_eigenvector = np.mean([eigenvector[n] for n in active_nodes]) if active_nodes else 0
        
        # Average path length
        try:
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G, weight='weight')
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
        except:
            avg_path_length = 0
        
        # Network centralization (Freeman's centralization index)
        centralization = self._freeman_centralization(G)
        
        metrics = {
            'density': density,
            'clustering_coefficient': clustering,
            'avg_betweenness_centrality': avg_betweenness,
            'avg_eigenvector_centrality': avg_eigenvector,
            'avg_path_length': avg_path_length,
            'centralization': centralization
        }
        
        # Normalize metrics before returning
        metrics = self._normalize_metrics(metrics, G)
        
        return metrics
    
    def _freeman_centralization(self, G):
        """Calculate Freeman's centralization index"""
        degree_centrality = nx.degree_centrality(G)
        if not degree_centrality:
            return 0
        
        max_centrality = max(degree_centrality.values())
        sum_differences = sum(max_centrality - c for c in degree_centrality.values())
        n = len(degree_centrality)
        max_possible_sum = (n - 1) * (n - 2) / (n - 1) if n > 2 else 0
        
        return sum_differences / max_possible_sum if max_possible_sum > 0 else 0
    
    def _normalize_metrics(self, metrics, G):
        """Normalize metrics for comparison"""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Normalize path length by theoretical minimum
        if metrics['avg_path_length'] > 0:
            metrics['normalized_path_length'] = 1 / metrics['avg_path_length']
        else:
            metrics['normalized_path_length'] = 0
        
        # Normalize centralization properly (Freeman's formula)
        degree_centrality = nx.degree_centrality(G)
        if degree_centrality and n_nodes > 2:
            max_possible_centralization = (n_nodes - 1) / (n_nodes - 2)
            metrics['normalized_centralization'] = metrics['centralization'] / max_possible_centralization if max_possible_centralization > 0 else 0
        else:
            metrics['normalized_centralization'] = 0
        
        return metrics
    
    def get_aggregated_results(self) -> pd.DataFrame:
        """Combine results from all matches"""
        if not self.results:
            return pd.DataFrame()
        
        all_results = pd.concat(self.results.values(), ignore_index=True)
        return all_results