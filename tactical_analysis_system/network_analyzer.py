import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List

class NetworkAnalyzer:
    """Calculates network metrics for contextual analysis"""
    
    def __init__(self):
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
                'centralization': 0
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
        
        # Network centralization (based on degree centrality)
        degree_centrality = nx.degree_centrality(G)
        max_centrality = max(degree_centrality.values()) if degree_centrality else 0
        centralization = max_centrality - np.mean(list(degree_centrality.values())) if degree_centrality else 0
        
        return {
            'density': density,
            'clustering_coefficient': clustering,
            'avg_betweenness_centrality': avg_betweenness,
            'avg_eigenvector_centrality': avg_eigenvector,
            'avg_path_length': avg_path_length,
            'centralization': centralization
        }
    
    def get_aggregated_results(self) -> pd.DataFrame:
        """Combine results from all matches"""
        if not self.results:
            return pd.DataFrame()
        
        all_results = pd.concat(self.results.values(), ignore_index=True)
        return all_results
