import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .utils import map_coordinates_to_zone

class NetworkBuilder:
    """
    Builds spatial passing networks from context-specific time windows.
    
    This class constructs undirected, weighted networks representing passing patterns
    between spatial zones on the football pitch. Networks are built for specific
    match contexts (e.g., winning/losing, early/late game) to enable comparative
    analysis of tactical behavior across different game states.
    
    The network construction process:
    1. Maps continuous pitch coordinates to discrete 7x7 grid zones (49 zones total)
    2. Aggregates passes between zones within each context window
    3. Creates undirected networks with normalized edge weights
    4. Maintains consistent node structure across all networks for comparability
    
    Attributes
    ----------
    networks : dict
        Storage for constructed networks (currently unused but available for caching).
    
    Notes
    -----
    **Key Methodological Choices:**
    
    - **Undirected Networks**: The implementation treats passing networks as undirected,
      meaning passes from zone A→B and B→A contribute to the same edge. This focuses
      on bilateral connectivity and zone-to-zone interaction patterns rather than
      directional flow, simplifying analysis while preserving essential structural
      properties for tactical evaluation.
    
    - **Normalized Edge Weights**: Edge weights are normalized by the total number of
      passes in each window (weight = pass_count / total_passes). This ensures networks
      are comparable across time windows with different passing volumes, which is
      critical for contextual analysis (RQ1).
    
    - **Consistent Node Structure**: All 49 zones are added as nodes regardless of
      whether passes occur in/through them. This maintains structural consistency
      across networks, enabling valid comparisons of network metrics.
    
    """
    
    def __init__(self):
        """
        Initialize NetworkBuilder.
        
        Creates an empty dictionary for potential network caching, though
        current implementation builds networks on-demand without caching.
        """
        self.networks = {}
    
    def build_networks_from_windows(self, context_windows: List[Dict]) -> List[Dict]:
        """
        Build passing networks for multiple context windows.
        
        Processes a list of context windows and constructs a spatial passing network
        for each window. Each window represents a specific time period and match
        context (e.g., minutes 10-20 while leading).
        
        Parameters
        ----------
        context_windows : list of dict
            List of context window dictionaries. Each window should contain:
            - 'passes': list of pass event dictionaries with location data
            - Additional context metadata (match_id, score context, phase, etc.)
            
            Expected pass event structure:
            {
                'location': [x, y],  # Origin coordinates
                'pass_end_location': [x, y],  # Destination coordinates
                'period': int,  # Match period (1 or 2)
                ... other event data
            }
        
        Returns
        -------
        list of dict
            List of dictionaries, each containing:
            - All original window metadata (unpacked from input)
            - 'network': nx.Graph object representing the passing network
            
            Windows with no valid passes (after filtering) are excluded from output.
        
        Notes
        -----
        - Windows without passes or with only invalid passes return None and are
          excluded from the output
        - Each network is built independently; no cross-window dependencies
        - Preserves all original window metadata for downstream analysis

        """
        network_data = []
        
        for window in context_windows:
            network = self._build_window_network(window)
            if network:
                network_data.append({
                    **window,  # Unpack all original window metadata
                    'network': network  # Add constructed network
                })
        
        return network_data
    
    def _build_window_network(self, window: Dict) -> nx.Graph:
        """
        Build a passing network for a single context window.
        
        Constructs an undirected, weighted network from pass events within a specific
        time window. Maps pass coordinates to spatial zones, aggregates passes between
        zones, and creates a network with normalized edge weights.
        
        Parameters
        ----------
        window : dict
            Context window dictionary containing:
            - 'passes': list of pass event dictionaries with location data
            - Other metadata (not used in network construction)
        
        Returns
        -------
        nx.Graph or None
            Undirected network with:
            - Nodes: Zone IDs (0-48) representing 7x7 grid positions
            - Edges: Connections between zones with attributes:
                - 'weight': Normalized weight (pass_count / total_passes)
                - 'raw_weight': Absolute number of passes between zones
            
            Returns None if:
            - No passes in window
            - All passes have missing/invalid location data
            - All passes are within the same zone (self-passes)
        
        Notes
        -----
        Processing steps:
        1. Convert passes list to DataFrame for efficient processing
        2. Map origin and destination coordinates to zones (accounting for period)
        3. Remove passes with missing zone data
        4. Create network with normalized edge weights
        
        """
        passes = window['passes']
        if not passes:
            return None
        
        df = pd.DataFrame(passes)
        
        # Map coordinates to zones with period information
        # Period is needed because coordinates are inverted in 2nd half
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
        # This filters out passes with invalid/missing coordinates
        df = df.dropna(subset=['origin_zone', 'dest_zone'])
        
        if len(df) == 0:
            return None
        
        return self._create_network(df)
    
    def _extract_zone_from_location(self, location, period):
        """
        Extract zone ID from coordinate location data.
        
        Wrapper function that validates location data and calls the utility
        function to map coordinates to the 7x7 zone grid.
        
        Parameters
        ----------
        location : list or None
            Coordinate pair [x, y] representing position on pitch.
            Expected format: [float, float] with x in [0, 120] and y in [0, 80].
        period : int
            Match period (1 for first half, 2 for second half).
            Used to normalize attack direction (coordinates inverted in period 2).
        
        Returns
        -------
        int or None
            Zone ID (0-48) if location is valid, None otherwise.
            
            Zone numbering (7x7 grid):
            - 0: Bottom-left (defensive left corner)
            - 6: Bottom-right (defensive right corner)
            - 42: Top-left (attacking left corner)
            - 48: Top-right (attacking right corner)
        
        Notes
        -----
        Returns None if:
        - location is None
        - location has fewer than 2 elements
        - coordinates are NaN (handled by map_coordinates_to_zone)

        """
        if not location or len(location) < 2:
            return None
        
        x, y = location[0], location[1]
        return map_coordinates_to_zone(x, y, period)
    
    def _create_network(self, passes: pd.DataFrame) -> nx.Graph:
        """
        Create weighted, undirected network from zone-mapped passes.
        
        Constructs a NetworkX graph where nodes represent spatial zones and edges
        represent passing connections between zones. Edge weights are normalized
        by total passes to enable cross-window comparisons.
        
        **Critical Methodological Note**: This is a key function for RQ1 analysis,
        as the normalization strategy directly affects how network metrics are
        compared across different match contexts.
        
        Parameters
        ----------
        passes : pd.DataFrame
            DataFrame of passes with columns:
            - 'origin_zone': int, zone ID where pass originated (0-48)
            - 'dest_zone': int, zone ID where pass ended (0-48)
            - Other columns are ignored
        
        Returns
        -------
        nx.Graph
            Undirected network with:
            
            **Nodes** (49 total):
            - All zones 0-48, regardless of whether passes occurred
            - Isolated nodes (no edges) have degree 0
            
            **Edges**:
            - Only between zones with at least one pass
            - Attributes:
                - 'weight': float, normalized weight = raw_weight / total_passes
                - 'raw_weight': int, absolute number of passes between zones
            
            **Self-loops**: Excluded (passes within same zone are ignored)
        
        Notes
        -----
        **Normalization Rationale**:
        Normalizing by total passes ensures that:
        1. Networks from different time windows are comparable
        2. Differences in passing volume don't confound structural analysis
        3. Edge weights represent relative importance within each context
        
        **Undirected Design**:
        Passes A→B and B→A contribute to the same edge. This:
        1. Focuses on bilateral connectivity patterns
        2. Simplifies network analysis (fewer edges, clearer structure)
        3. Preserves essential tactical relationships
        
        **All Nodes Included**:
        Including all 49 zones ensures:
        1. Consistent network structure across all windows
        2. Valid comparisons of centrality measures
        3. Ability to identify unused zones (tactical insights)
        
        """
        G = nx.Graph()
        
        # Add all possible zones as nodes (0-48 for 7x7 grid)
        # This ensures consistent structure across all networks
        G.add_nodes_from(range(49))
        
        # Count passes between zones
        edge_weights = {}
        total_passes = 0
        
        for _, pass_event in passes.iterrows():
            origin = int(pass_event['origin_zone'])
            dest = int(pass_event['dest_zone'])
            
            # Exclude self-passes (passes within the same zone)
            if origin != dest:
                # Create undirected edge by sorting node IDs
                # This ensures (A,B) and (B,A) map to the same edge
                edge = tuple(sorted([origin, dest]))
                edge_weights[edge] = edge_weights.get(edge, 0) + 1
                total_passes += 1

        # Normalize edge weights by total number of passes in the window
        # This is critical for cross-context comparisons (RQ1)
        for (node1, node2), weight in edge_weights.items():
            norm_weight = weight / total_passes if total_passes > 0 else 0
            G.add_edge(
                node1, 
                node2, 
                weight=norm_weight,      # Normalized for comparisons
                raw_weight=weight        # Preserved for reference
            )
        
        return G
