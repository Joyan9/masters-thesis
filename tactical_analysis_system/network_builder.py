import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import map_coordinates_to_zone

class NetworkBuilder:
    """Builds passing networks from event data"""
    
    def __init__(self):
        self.networks = {}
    
    def build_contextual_networks(self, events: List[Dict], contexts: Dict, 
                                 match_id: str) -> Dict:
        """Build networks for each context period"""
        df = pd.DataFrame(events)
        passes = df[df['type'] == 'Pass'].copy()
        
        # Map coordinates to zones
        passes['origin_zone'] = passes.apply(
            lambda x: map_coordinates_to_zone(x.get('location', [None, None])[0] if x.get('location') else None,
                                            x.get('location', [None, None])[1] if x.get('location') else None), 
            axis=1
        )
        passes['dest_zone'] = passes.apply(
            lambda x: map_coordinates_to_zone(x.get('pass_end_location', [None, None])[0] if x.get('pass_end_location') else None,
                                            x.get('pass_end_location', [None, None])[1] if x.get('pass_end_location') else None), 
            axis=1
        )
        
        # Remove passes with missing zone data
        passes = passes.dropna(subset=['origin_zone', 'dest_zone'])
        passes['minute'] = passes['minute'].fillna(0)
        
        networks = {}
        
        # Build networks for each context type
        for context_type, periods in contexts.items():
            networks[context_type] = {}
            
            for start_min, end_min, context_label in periods:
                # Filter passes in this period
                period_passes = passes[
                    (passes['minute'] >= start_min) & 
                    (passes['minute'] < end_min)
                ]
                
                if len(period_passes) > 0:
                    network = self._create_network(period_passes)
                    networks[context_type][context_label] = {
                        'network': network,
                        'period': (start_min, end_min),
                        'pass_count': len(period_passes)
                    }
        
        self.networks[match_id] = networks
        return networks
    
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
