import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

# Configure logging for error tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class NetworkAnalyzer:
    """
    Calculates network metrics for contextual passing network analysis.
    
    This class computes graph-theoretic metrics on spatial passing networks to
    quantify tactical patterns. It calculates both node-level centrality measures
    (aggregated to network-level averages) and graph-level structural properties.
    
    These metrics enable RQ1 (Contextual Network Analysis) by providing quantitative
    measures of network structure that can be compared across different match contexts
    (score states, match phases, intensity levels).
    
    Attributes
    ----------
    data_loader : DataLoader
        Reference to data loader (for potential future use).
    results : dict
        Storage for calculated metrics (currently unused but available for caching).
    
    Notes
    -----
    **Key Metrics Calculated:**
    
    1. **Density**: Proportion of actual edges to possible edges
       - Range: [0, 1]
       - Interpretation: How interconnected the passing network is
    
    2. **Clustering Coefficient**: Tendency of zones to form triangular patterns
       - Range: [0, 1]
       - Interpretation: Local cohesion in passing patterns
    
    3. **Betweenness Centrality**: Extent to which zones act as bridges
       - Averaged across active nodes
       - Interpretation: Importance of zones in connecting different areas
    
    4. **Eigenvector Centrality**: Influence based on connections to important zones
       - Averaged across active nodes
       - Interpretation: Strategic importance in network structure
    
    5. **Average Path Length**: Mean shortest path between zone pairs
       - Lower = more direct connectivity
       - Interpretation: Efficiency of ball movement across pitch
    
    6. **Centralization (Freeman's)**: Degree to which network is organized around hubs
       - Range: [0, 1]
       - Interpretation: Hierarchical vs. distributed structure
    
    **Critical Methodological Note - Weight Interpretation:**
    
    Edge weights represent normalized pass frequencies (higher = more passes).
    However, NetworkX centrality and path algorithms interpret weights as DISTANCES
    (higher = farther apart). Therefore, weights are INVERTED (1/weight) for:
    - Betweenness centrality
    - Eigenvector centrality  
    - Average path length
    
    This ensures that zones with more passes between them are treated as "closer"
    in the network topology, which aligns with tactical interpretation.
    
    """
    
    def __init__(self, data_loader):
        """
        Initialize NetworkAnalyzer.
        
        Parameters
        ----------
        data_loader : DataLoader
            Reference to the data loader instance. Currently stored for potential
            future use (e.g., accessing match metadata for context-aware analysis).
        """
        self.data_loader = data_loader
        self.results = {}
    
    def _calculate_network_metrics(self, G: nx.Graph) -> Dict:
        """
        Calculate comprehensive network metrics for a passing network.
        
        Computes both node-level centrality measures (aggregated to network averages)
        and graph-level structural properties. Handles edge cases like empty networks,
        disconnected components, and convergence failures.
        
        Parameters
        ----------
        G : nx.Graph
            Undirected, weighted passing network with:
            - Nodes: Zone IDs (0-48)
            - Edges: Connections with 'weight' attribute (normalized pass frequency)
            - May contain isolated nodes (zones with no passes)
        
        Returns
        -------
        dict
            Dictionary of network metrics:
            - 'density': float [0, 1]
            - 'clustering_coefficient': float [0, 1]
            - 'avg_betweenness_centrality': float [0, 1]
            - 'avg_eigenvector_centrality': float [0, 1]
            - 'avg_path_length': float ≥ 0
            - 'centralization': float [0, 1]
            - 'normalized_path_length': float [0, 1] (inverse of avg_path_length)
            
            All metrics return 0 for empty networks (no edges).
        
        Notes
        -----
        **Weight Inversion for Centrality/Path Calculations:**
        
        Edge weights represent pass frequencies (higher = more passes = stronger connection).
        But NetworkX treats weights as distances (higher = farther = weaker connection).
        
        To align with tactical interpretation, we invert weights:
        - Original weight: 0.3 (30% of passes in window)
        - Inverted weight: 1/0.3 = 3.33 (distance metric)
        
        This ensures:
        - Zones with frequent passing are treated as "close"
        - Betweenness identifies zones bridging different passing clusters
        - Path length reflects tactical distance, not just topological hops
        
        **Active Nodes Filter:**
        
        Centrality averages exclude isolated nodes (degree = 0) to avoid:
        - Diluting metrics with meaningless zeros
        - Misrepresenting the structure of the active passing network
        
        This is standard practice in network analysis when nodes can be structurally
        absent from the phenomenon being studied.
        
        **Error Handling:**
        
        - Eigenvector centrality: May fail to converge; falls back to zeros with warning
        - Path length: Handles disconnected graphs by using largest component
        """
        # Handle empty networks
        if G.number_of_edges() == 0:
            return {
                'density': 0,
                'clustering_coefficient': 0,
                'avg_betweenness_centrality': 0,
                'avg_eigenvector_centrality': 0,
                'avg_path_length': 0,
                'centralization': 0,
                'normalized_path_length': 0
            }
        
        # Basic structural metrics
        density = nx.density(G)
        clustering = nx.average_clustering(G, weight='weight')
        
        # Prepare inverted weights for centrality/path calculations
        # Original weights = pass frequencies (higher = stronger connection)
        # Inverted weights = distances (higher = weaker connection)
        # This aligns with NetworkX's interpretation of weights as distances
        G_inverted = G.copy()
        for u, v, data in G_inverted.edges(data=True):
            if 'weight' in data and data['weight'] > 0:
                data['weight'] = 1.0 / data['weight']
            else:
                # Handle edge case of zero weight (shouldn't occur with normalized weights)
                data['weight'] = float('inf')
        
        # Centrality measures (using inverted weights)
        betweenness = nx.betweenness_centrality(G_inverted, weight='weight')
        
        # Eigenvector centrality may fail to converge
        try:
            eigenvector = nx.eigenvector_centrality(G_inverted, weight='weight', max_iter=1000)
        except nx.PowerIterationFailedConvergence as e:
            logger.warning(f"Eigenvector centrality failed to converge: {e}. Using zeros.")
            eigenvector = {node: 0 for node in G.nodes()}
        except Exception as e:
            logger.error(f"Unexpected error in eigenvector centrality: {e}. Using zeros.")
            eigenvector = {node: 0 for node in G.nodes()}
        
        # Average centralities (only for active nodes to avoid dilution)
        # Active nodes = nodes with at least one edge
        active_nodes = [n for n in G.nodes() if G.degree(n) > 0]
        
        avg_betweenness = np.mean([betweenness[n] for n in active_nodes]) if active_nodes else 0
        avg_eigenvector = np.mean([eigenvector[n] for n in active_nodes]) if active_nodes else 0
        
        # Average path length (using inverted weights)
        try:
            if nx.is_connected(G_inverted):
                avg_path_length = nx.average_shortest_path_length(G_inverted, weight='weight')
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(G_inverted), key=len)
                subgraph = G_inverted.subgraph(largest_cc)
                if len(subgraph) > 1:
                    avg_path_length = nx.average_shortest_path_length(subgraph, weight='weight')
                else:
                    avg_path_length = 0
        except Exception as e:
            logger.warning(f"Error calculating average path length: {e}. Using 0.")
            avg_path_length = 0
        
        # Network centralization (Freeman's degree centralization)
        centralization = self._freeman_centralization(G)
        
        # Compile metrics
        metrics = {
            'density': density,
            'clustering_coefficient': clustering,
            'avg_betweenness_centrality': avg_betweenness,
            'avg_eigenvector_centrality': avg_eigenvector,
            'avg_path_length': avg_path_length,
            'centralization': centralization
        }
        
        # Add derived metrics
        metrics = self._add_derived_metrics(metrics, G)
        
        return metrics
    
    def _freeman_centralization(self, G: nx.Graph) -> float:
        """
        Calculate Freeman's degree centralization index.
        
        Freeman's centralization quantifies the extent to which a network is organized
        around central "hub" nodes. It compares the actual degree distribution to the
        most centralized possible structure (a star graph).
        
        Formula (Freeman, 1978):
        C_D = Σ[C_D(n*) - C_D(n_i)] / [(n-1)(n-2)]
        
        Where:
        - C_D(n*) = maximum degree centrality in the network
        - C_D(n_i) = degree centrality of node i
        - n = number of nodes
        - Denominator = maximum possible sum of differences (star graph)
        
        Parameters
        ----------
        G : nx.Graph
            Network to analyze.
        
        Returns
        -------
        float
            Centralization index in range [0, 1]:
            - 0: Completely decentralized (e.g., complete graph, all nodes equal)
            - 1: Completely centralized (e.g., star graph, one hub dominates)
        
        Notes
        -----
        **Interpretation for Passing Networks:**
        
        - High centralization (→1): Passing flows through few key zones (hub-based tactics)
        - Low centralization (→0): Passing distributed across many zones (fluid tactics)
        
        **Reference:**
        Freeman, L. C. (1978). Centrality in social networks conceptual clarification.
        Social Networks, 1(3), 215-239.
        
        The formula normalizes by the maximum possible centralization (star graph with
        n nodes), making values comparable across networks of different sizes.
        
        """
        degree_centrality = nx.degree_centrality(G)
        
        if not degree_centrality:
            return 0
        
        n = len(degree_centrality)
        
        # Need at least 3 nodes for meaningful centralization
        if n < 3:
            return 0
        
        # Find maximum degree centrality
        max_centrality = max(degree_centrality.values())
        
        # Sum of differences from maximum
        sum_differences = sum(max_centrality - c for c in degree_centrality.values())
        
        # Maximum possible sum of differences (occurs in star graph)
        # For star graph: one node has centrality (n-1)/(n-1) = 1
        #                 other nodes have centrality 1/(n-1)
        # Sum of differences = (n-1) * [1 - 1/(n-1)] = (n-1) * [(n-2)/(n-1)] = (n-2)
        # But Freeman's formula uses unnormalized degrees, giving (n-1)(n-2)
        max_possible_sum = (n - 1) * (n - 2)
        
        # Calculate centralization
        centralization = sum_differences / max_possible_sum if max_possible_sum > 0 else 0
        
        return centralization
    
    def _add_derived_metrics(self, metrics: Dict, G: nx.Graph) -> Dict:
        """
        Add derived/normalized metrics to the metrics dictionary.
        
        Creates additional metrics that may be useful for analysis, such as
        inverse transformations or normalized versions.
        
        Parameters
        ----------
        metrics : dict
            Dictionary of base metrics from _calculate_network_metrics.
        G : nx.Graph
            Network being analyzed (for context).
        
        Returns
        -------
        dict
            Updated metrics dictionary with additional derived metrics.
        
        Notes
        -----
        **Normalized Path Length:**
        
        Calculated as the inverse of average path length (1 / avg_path_length).
        This transformation makes the metric positively oriented:
        - Higher values = shorter paths = better connectivity
        - Aligns with other metrics where higher = more connected
        
        Useful for correlation analysis and visualization where consistent
        directionality is desired.
        
        **Normalized Centralization:**
        
        Freeman's centralization is already normalized to [0, 1] range by design,
        so no additional normalization is needed. This metric is included for
        completeness but equals the base centralization value.
        """
        # Normalize path length by inversion
        # Higher normalized value = shorter paths = better connectivity
        if metrics['avg_path_length'] > 0:
            metrics['normalized_path_length'] = 1.0 / metrics['avg_path_length']
        else:
            metrics['normalized_path_length'] = 0
        
        # Note: Freeman's centralization is already normalized to [0, 1]
        # No additional normalization needed - the formula includes normalization
        # by the maximum possible centralization (star graph)
        # Including this for documentation purposes, but it's redundant
        # metrics['normalized_centralization'] = metrics['centralization']
        
        return metrics
    
    def get_aggregated_results(self) -> pd.DataFrame:
        """
        Combine results from all analyzed matches.
        
        Aggregates network metrics across all matches that have been analyzed.
        Currently not used in the main RQ1 pipeline (which processes results
        directly), but available for batch analysis workflows.
        
        Returns
        -------
        pd.DataFrame
            Combined DataFrame of all network metrics from all matches.
            Empty DataFrame if no results have been stored.
        
        Notes
        -----
        This method is available for future use but is not part of the current
        RQ1 analysis pipeline, which processes metrics directly from individual
        networks without storing intermediate results.
        """
        if not self.results:
            return pd.DataFrame()
        
        all_results = pd.concat(self.results.values(), ignore_index=True)
        return all_results
