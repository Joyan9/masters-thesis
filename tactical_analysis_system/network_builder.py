import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import map_coordinates_to_zone

class NetworkBuilder:
    """Builds passing networks from context windows"""
    
    def __init__(self):
        self.networks = {}
    
    def build_networks_from_windows(self, context_windows: List[Dict]) -> List[Dict]:
        """Build networks for each context window"""
        network_data = []
        
        for window in context_windows:
            network = self._build_window_network(window)
            if network:
                network_data.append({
                    **window,
                    'network': network
                })
        
        return network_data
    
    def _build_window_network(self, window: Dict) -> nx.Graph:
        """Build network for a single context window"""
        passes = window['passes']
        if not passes:
            return None
        
        df = pd.DataFrame(passes)
        
        # Map coordinates to zones with period information
        df['origin_zone'] = df.apply(
            lambda row: self._extract_zone_from_location(
                row.get('location'), row.get('period')
            ), axis=1
        )
        
        df['dest_zone'] = df.apply(
            lambda row: self._extract_zone_from_location(
                row.get('pass_end_location'), row.get('period')
            ), axis=1
        )
        
        # Remove passes with missing zone data
        df = df.dropna(subset=['origin_zone', 'dest_zone'])
        
        if len(df) == 0:
            return None
        
        return self._create_network(df)
    
    def _extract_zone_from_location(self, location, period):
        """Extract zone from location data"""
        if not location or len(location) < 2:
            return None
        
        x, y = location[0], location[1]
        return map_coordinates_to_zone(x, y, period)
    
    def _create_network(self, passes: pd.DataFrame) -> nx.Graph:
        """Create weighted network from passes"""
        G = nx.Graph()
        
        # Add all possible zones as nodes (0-48 for 7x7 grid)
        G.add_nodes_from(range(49))
        
        # Count passes between zones
        edge_weights = {}
        for _, pass_event in passes.iterrows():
            origin = int(pass_event['origin_zone'])
            dest = int(pass_event['dest_zone'])
            
            if origin != dest:  # Exclude self-passes
                edge = tuple(sorted([origin, dest]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1
        
        # Add weighted edges
        for (node1, node2), weight in edge_weights.items():
            G.add_edge(node1, node2, weight=weight)
        
        return G
