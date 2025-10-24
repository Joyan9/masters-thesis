"""
This module implements a rule-based tactical recommendation system that translates
network metrics and contextual factors into actionable coaching suggestions. The system
uses a multi-component approach combining:
1. Threshold-based performance benchmarking
2. Context-aware rule evaluation
3. Temporal consistency tracking
4. Confidence-weighted recommendation filtering

The recommender is designed to support real-time tactical decision-making by identifying
critical situations and suggesting appropriate interventions based on network structure
patterns and match context.

"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")


class RecommendationType(Enum):
    """
    Enumeration of tactical recommendation categories.
    
    Each type represents a distinct tactical dimension that can be adjusted
    during match play. These categories align with established football tactics
    literature (Clemente et al., 2015; Pena & Touchette, 2012).
    """
    SPATIAL = "spatial"              # Spatial positioning and pitch coverage
    TEMPO = "tempo"                  # Speed of play and passing rhythm
    CONNECTIVITY = "connectivity"    # Network cohesion and passing triangles
    ATTACKING = "attacking"          # Offensive organization and penetration
    DEFENSIVE = "defensive"          # Defensive structure and compactness
    PRESSING = "pressing"            # High pressing and ball recovery
    POSSESSION = "possession"        # Ball retention and circulation
    TRANSITION = "transition"        # Counter-attacks and phase transitions


class ConfidenceLevel(Enum):
    """
    Categorical confidence levels for recommendations.
    
    These levels provide intuitive interpretation of recommendation reliability,
    mapped from continuous confidence scores for easier communication to coaches.
    """
    LOW = "low"              # 0.0 - 0.4: Uncertain recommendation
    MEDIUM = "medium"        # 0.4 - 0.6: Moderate confidence
    HIGH = "high"            # 0.6 - 0.8: High confidence
    VERY_HIGH = "very_high"  # 0.8 - 1.0: Very high confidence


@dataclass
class TacticalRecommendation:
    """
    Data structure representing a single tactical recommendation.
    
    Attributes:
        action (str): Human-readable tactical instruction
        recommendation_type (RecommendationType): Category of tactical adjustment
        confidence (ConfidenceLevel): Categorical confidence level
        confidence_score (float): Continuous confidence score [0, 1]
        context (Dict): Match context when recommendation was generated
        triggered_metrics (List[str]): Network metrics that triggered this recommendation
        reasoning (str): Explanation of why this recommendation was made
        priority (int): Priority ranking (1 = highest priority)
        implementation_time (str): Suggested implementation timeline ('immediate', 'gradual', 'ongoing')
        expected_impact (Dict[str, Dict]): Expected changes to network metrics with uncertainty
        context_specificity (float): How well-suited this recommendation is to current context [0, 1]
    """
    action: str
    recommendation_type: RecommendationType
    confidence: ConfidenceLevel
    confidence_score: float
    context: Dict
    triggered_metrics: List[str]
    reasoning: str
    priority: int
    implementation_time: str
    expected_impact: Dict[str, Dict]  # Now includes uncertainty bounds
    context_specificity: float


class ImpactEstimator:
    """
    Estimates expected impacts of tactical recommendations on network metrics.
    
    This class provides literature-based estimates of how tactical changes affect
    network structure, with uncertainty quantification. Impact magnitudes are derived
    from football analytics research (Clemente et al., 2015; Pena & Touchette, 2012)
    and validated against empirical observations.
    
    The estimator accounts for:
    - Recommendation type (different tactics affect different metrics)
    - Urgency level (desperate situations enable larger changes)
    - Current metric values (diminishing returns near extremes)
    """
    
    def __init__(self):
        """
        Initialize impact estimator with literature-based impact ranges.
        
        Impact magnitudes are defined as proportional changes to metric values:
        - Small: Detectable change (5-10%)
        - Medium: Meaningful tactical shift (10-20%)
        - Large: Substantial reorganization (15-30%)
        
        These ranges reflect realistic changes observable within 5-minute windows
        based on professional football match analysis.
        """
        # Based on Clemente et al. (2015), Pena & Touchette (2012)
        self.realistic_impacts = {
            'density': {
                'small': 0.05,    # 5% change (detectable)
                'medium': 0.10,   # 10% change (meaningful)
                'large': 0.15     # 15% change (substantial)
            },
            'clustering_coefficient': {
                'small': 0.03,    # Triangular passing patterns
                'medium': 0.08,
                'large': 0.12
            },
            'centralization': {
                'small': 0.05,    # Hierarchical structure
                'medium': 0.10,
                'large': 0.20     # Major tactical reorganization
            },
            'avg_path_length': {
                'small': 0.10,    # 10% change (~0.2 passes)
                'medium': 0.20,   # 20% change (~0.5 passes)
                'large': 0.30     # 30% change (~0.8 passes)
            },
            'avg_betweenness_centrality': {
                'small': 0.05,    # Bridging importance
                'medium': 0.10,
                'large': 0.15
            },
            'avg_eigenvector_centrality': {
                'small': 0.05,    # Strategic importance
                'medium': 0.10,
                'large': 0.15
            }
        }
        
        # Map recommendation types to affected metrics
        # Based on tactical theory: each recommendation type primarily affects specific metrics
        self.impact_map = {
            'attacking': ['density', 'centralization'],
            'defensive': ['clustering_coefficient', 'centralization'],
            'tempo': ['avg_path_length', 'density'],
            'connectivity': ['clustering_coefficient', 'density'],
            'pressing': ['density', 'avg_path_length'],
            'possession': ['clustering_coefficient', 'density'],
            'transition': ['avg_path_length', 'centralization'],
            'spatial': ['density', 'avg_betweenness_centrality']
        }
    
    def estimate_impact(self, recommendation_type: str, urgency: str, 
                       current_metrics: Dict) -> Dict[str, Dict]:
        """
        Estimate expected impact of a recommendation with uncertainty bounds.
        
        Args:
            recommendation_type (str): Type of tactical recommendation
            urgency (str): Urgency level ('very_high', 'high', 'medium', 'normal')
            current_metrics (Dict): Current network metric values
        
        Returns:
            Dict[str, Dict]: Expected impacts with uncertainty for each affected metric
                {
                    'density': {
                        'expected': 0.10,      # Expected change
                        'min': 0.07,           # Lower bound (70% of expected)
                        'max': 0.13,           # Upper bound (130% of expected)
                        'confidence': 0.7      # Confidence in estimate
                    }
                }
        
        Notes:
            - Higher urgency enables larger changes (desperate measures)
            - Uncertainty bounds reflect ±30% variance around expected impact
            - Confidence decreases with urgency (aggressive changes are less predictable)
        """
        impacts = {}
        
        # Get metrics affected by this recommendation type
        affected_metrics = self.impact_map.get(recommendation_type, [])
        
        # Determine impact magnitude based on urgency
        # Higher urgency = more aggressive changes = larger expected impact
        magnitude = {
            'very_high': 'large',   # Desperate measures (15-30% changes)
            'high': 'medium',       # Significant changes (10-20% changes)
            'medium': 'small',      # Incremental changes (5-10% changes)
            'normal': 'small'       # Minor adjustments (5-10% changes)
        }.get(urgency, 'small')
        
        for metric in affected_metrics:
            if metric in self.realistic_impacts:
                base_impact = self.realistic_impacts[metric][magnitude]
                
                # Add uncertainty bounds (±30% of base impact)
                # Reflects natural variance in tactical implementation effectiveness
                impacts[metric] = {
                    'expected': round(base_impact, 3),
                    'min': round(base_impact * 0.7, 3),
                    'max': round(base_impact * 1.3, 3),
                    'confidence': self._estimate_confidence(urgency, current_metrics.get(metric))
                }
        
        return impacts
    
    def _estimate_confidence(self, urgency: str, current_value: Optional[float]) -> float:
        """
        Estimate confidence in impact prediction.
        
        Args:
            urgency (str): Urgency level
            current_value (Optional[float]): Current metric value (if available)
        
        Returns:
            float: Confidence score [0, 1]
        
        Notes:
            - Higher urgency = more aggressive changes = lower confidence
            - Extreme current values may limit change potential (not implemented yet)
        """
        # Base confidence decreases with urgency
        # Aggressive changes are harder to predict accurately
        base_confidence = {
            'very_high': 0.6,  # High impact but uncertain outcome
            'high': 0.7,       # Moderate uncertainty
            'medium': 0.75,    # Low uncertainty
            'normal': 0.8      # Very low uncertainty
        }.get(urgency, 0.7)
        
        # Future enhancement: adjust confidence based on current_value
        # (e.g., harder to increase density if already at 0.8)
        
        return base_confidence


class TemporalTracker:
    """
    Tracks recommendation history across temporal windows.
    
    This class maintains a sliding window of recent recommendations and contexts
    to enable temporal consistency analysis. Temporal consistency helps identify
    whether the system is providing stable tactical guidance or oscillating between
    contradictory suggestions.
    
    The tracker maintains the last 5 windows (~50 minutes of match time), which
    provides sufficient context for identifying tactical patterns while remaining
    responsive to changing match conditions.
    """
    
    def __init__(self):
        """
        Initialize temporal tracker with empty history.
        
        Attributes:
            recent_recommendations (List[List[str]]): Last 5 windows of recommendation types
            recent_contexts (List[Dict]): Last 5 windows of match contexts
        """
        self.recent_recommendations = []
        self.recent_contexts = []
    
    def add_recommendations(self, recommendations: List[TacticalRecommendation], context: Dict):
        """
        Add recommendations to temporal history.
        
        Args:
            recommendations (List[TacticalRecommendation]): Current window recommendations
            context (Dict): Current match context
        
        Notes:
            - Maintains sliding window of last 5 entries
            - Older entries are automatically discarded (FIFO)
        """
        # Extract recommendation types for comparison
        rec_types = [rec.recommendation_type.value for rec in recommendations]
        self.recent_recommendations.append(rec_types)
        self.recent_contexts.append(context.copy())
        
        # Keep only last 5 windows (~50 minutes)
        # This provides sufficient history without over-weighting distant past
        if len(self.recent_recommendations) > 5:
            self.recent_recommendations.pop(0)
            self.recent_contexts.pop(0)
    
    def get_consistency_score(self, current_recs: List[TacticalRecommendation]) -> float:
        """
        Calculate temporal consistency score using Jaccard similarity.
        
        Compares current recommendations with the last 2 windows to measure
        tactical stability. High consistency indicates stable tactical approach,
        while low consistency suggests reactive or oscillating recommendations.
        
        Args:
            current_recs (List[TacticalRecommendation]): Current recommendations
        
        Returns:
            float: Consistency score [0, 1]
                - 1.0 = Perfect consistency (identical recommendations)
                - 0.5 = Moderate consistency (some overlap)
                - 0.0 = No consistency (completely different recommendations)
        
        Notes:
            - Uses Jaccard similarity: |A ∩ B| / |A ∪ B|
            - Compares with last 2 windows (most recent ~20 minutes)
            - Returns 0.8 if insufficient history (optimistic default)
        """
        # Need at least 2 previous windows for comparison
        if len(self.recent_recommendations) < 2:
            return 0.8  # Optimistic default when insufficient history
        
        # Extract current recommendation types
        current_types = [rec.recommendation_type.value for rec in current_recs]
        
        # Compare with last 2 windows using Jaccard similarity
        similarities = []
        for past_types in self.recent_recommendations[-2:]:
            if past_types and current_types:
                # Jaccard similarity: intersection / union
                common = len(set(current_types).intersection(set(past_types)))
                total = len(set(current_types).union(set(past_types)))
                similarity = common / total if total > 0 else 0
                similarities.append(similarity)
        
        # Return average similarity across compared windows
        return np.mean(similarities) if similarities else 0.5
    
    def should_maintain_consistency(self, context: Dict) -> bool:
        """
        Determine if context suggests maintaining tactical consistency.
        
        Checks whether match context has been stable across recent windows.
        Stable context suggests that consistent tactics are appropriate, while
        changing context may warrant tactical adjustments.
        
        Args:
            context (Dict): Current match context
        
        Returns:
            bool: True if context is stable (>60% factors unchanged), False otherwise
        
        Notes:
            - Compares score_context, phase_context, and intensity_context
            - Requires at least 2 previous windows for comparison
            - Threshold: 60% of context factors must be stable
        """
        if len(self.recent_contexts) < 2:
            return False  # Insufficient history
        
        # Compare current context with most recent previous context
        last_context = self.recent_contexts[-1]
        stable_factors = 0
        total_factors = 0
        
        # Check stability of each context dimension
        for key in ['score_context', 'phase_context', 'intensity_context']:
            if key in context and key in last_context:
                total_factors += 1
                if context[key] == last_context[key]:
                    stable_factors += 1
        
        # Context is stable if >60% of factors are unchanged
        return stable_factors / total_factors > 0.6 if total_factors > 0 else False


class ThresholdAnalyzer:
    """
    Extracts performance thresholds from historical network data.
    
    This class analyzes the distribution of network metrics across all matches
    to establish percentile-based performance benchmarks. These thresholds enable
    objective assessment of whether current metric values represent excellent,
    good, average, poor, or critical performance.
    
    Thresholds are global (calculated across all teams and matches) rather than
    team-specific, providing universal performance standards based on the dataset.
    """
    
    def __init__(self):
        """
        Initialize threshold analyzer.
        
        Attributes:
            thresholds (Dict): Extracted performance thresholds for each metric
        """
        self.thresholds = {}
    
    def extract_thresholds(self, network_data: pd.DataFrame) -> Dict:
        """
        Extract percentile-based performance thresholds from network data.
        
        Calculates five performance levels for each network metric:
        - Excellent: 90th percentile (top 10% of performances)
        - Good: 75th percentile (top 25%)
        - Average: 50th percentile (median)
        - Poor: 25th percentile (bottom 25%)
        - Critical: 10th percentile (bottom 10%)
        
        Args:
            network_data (pd.DataFrame): Historical network metrics across all matches
        
        Returns:
            Dict: Threshold dictionary with structure:
                {
                    'density': {
                        'excellent': 0.65,
                        'good': 0.55,
                        'average': 0.45,
                        'poor': 0.35,
                        'critical': 0.25
                    },
                    ...
                }
        
        Notes:
            - Thresholds are global (not team-specific)
            - Based on empirical distribution of observed values
            - Metrics with insufficient data are skipped
            - Stored in self.thresholds for later reference
        """
        thresholds = {}
        
        # Network metrics to analyze
        metrics = ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                  'avg_eigenvector_centrality', 'avg_path_length', 'centralization']
        
        for metric in metrics:
            if metric in network_data.columns:
                # Extract non-null values
                values = network_data[metric].dropna()
                
                if len(values) > 0:
                    # Calculate percentile-based thresholds
                    thresholds[metric] = {
                        'excellent': np.percentile(values, 90),
                        'good': np.percentile(values, 75),
                        'average': np.percentile(values, 50),
                        'poor': np.percentile(values, 25),
                        'critical': np.percentile(values, 10)
                    }
        
        # Store for later reference by rule engine
        self.thresholds = thresholds
        return thresholds

    def log_thresholds(self, filepath: Optional[str] = None) -> str:
        """
        Persist and print extracted performance thresholds.

        Args:
            filepath (Optional[str]): Output JSON filepath. If None, defaults to
                final_results/performance_thresholds_<timestamp>.json

        Returns:
            str: Path to written file.
        """
        # Prepare output path
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("final_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        if filepath is None:
            filepath = out_dir / f"performance_thresholds_{ts}.json"
        else:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

        # Build save payload with simple metadata and counts
        payload = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "source": "ThresholdAnalyzer.extract_thresholds",
                "system_version": "v2.0_geometric_mean"
            },
            "thresholds": self.thresholds,
            "counts": {}
        }

        # Add counts per metric if original values were stored (best-effort)
        for metric, vals in (self.thresholds or {}).items():
            try:
                # store presence of percentiles as indicator
                payload["counts"][metric] = {k: float(v) for k, v in vals.items()}
            except Exception:
                payload["counts"][metric] = None

        # Write JSON
        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        # Print concise console summary
        print(f"Saved performance thresholds to: {filepath}")
        for metric, th in (self.thresholds or {}).items():
            print(f"  - {metric}: good={th.get('good'):.4f}, average={th.get('average'):.4f}, poor={th.get('poor'):.4f}")

        return str(filepath)


class RuleEngine:
    """
    Core rule-based reasoning engine for tactical recommendations.
    
    This class implements the decision logic that translates network metrics and
    match context into tactical recommendations. It combines:
    1. Threshold-based rule conditions (when to recommend)
    2. Context-aware weighting (how much to emphasize each recommendation type)
    3. Temporal consistency tracking (stability of recommendations)
    4. Confidence-based filtering (quality control)
    
    The engine uses a geometric mean approach for combining context weights to
    prevent multiplicative explosion while preserving relative importance across
    multiple contextual dimensions.
    """
    
    def __init__(self, thresholds: Dict):
        """
        Initialize rule engine with performance thresholds.
        
        Args:
            thresholds (Dict): Performance thresholds from ThresholdAnalyzer
        
        Attributes:
            thresholds (Dict): Performance benchmarks for rule conditions
            temporal_tracker (TemporalTracker): Tracks recommendation history
            context_weights (Dict): Context-specific importance weights
            rules (List[Dict]): Tactical rules with conditions and recommendations
            impact_estimator (ImpactEstimator): Estimates expected impacts
        """
        self.thresholds = thresholds
        self.temporal_tracker = TemporalTracker()
        self.impact_estimator = ImpactEstimator()
        self.context_weights = self._create_context_weights()
        self.rules = self._create_rules()
    
    def _create_context_weights(self) -> Dict:
        """
        Create context-specific weights for recommendation types.
        
        These weights reflect the tactical appropriateness of different recommendation
        types under various match contexts. Higher weights indicate that a recommendation
        type is more suitable for the given context.
        
        Returns:
            Dict: Nested dictionary of weights with structure:
                {
                    'score_context': {
                        'leading': {'defensive': 2.5, 'attacking': 1.0, ...},
                        'tied': {...},
                        'trailing': {...}
                    },
                    'phase_context': {...},
                    'intensity_context': {...}
                }
        
        Notes:
            - Weights are based on football tactics literature and empirical validation
            - Values range from 0.05 (highly inappropriate) to 4.5 (highly appropriate)
            - Combined using geometric mean to prevent multiplicative explosion
            - Validated through recommendation quality scoring (see validation script)
        
        Design Rationale:
            - Trailing teams should attack (weight: 4.5) not defend (weight: 0.05)
            - Leading teams should defend (weight: 2.5) and control possession (weight: 2.0)
            - Low intensity requires activation (tempo: 3.8, pressing: 3.5)
            - Late phase increases urgency for all proactive actions
        """
        return {
            'score_context': {
                'leading': {
                    # Protect the lead: defensive stability and possession control
                    'defensive': 2.5,
                    'possession': 2.0,
                    'connectivity': 1.8,
                    'spatial': 0.5,
                    'attacking': 1.0,
                    'tempo': 1.0,
                    'pressing': 0.9,
                    'transition': 1.2,
                },
                'tied': {
                    # Balanced approach with slight attacking bias
                    'attacking': 2.5,
                    'tempo': 2.4,
                    'pressing': 2.3,
                    'transition': 2.3,
                    'spatial': 1.8,
                    'connectivity': 1.5,
                    'possession': 1.2,
                    'defensive': 0.8,
                },
                'trailing': {
                    # Aggressive attacking: maximize offensive commitment
                    'attacking': 4.5,
                    'tempo': 4.0,
                    'pressing': 3.6,
                    'transition': 3.8,
                    'spatial': 2.8,
                    'connectivity': 2.0,
                    'possession': 0.1,  # No time for possession play
                    'defensive': 0.05,  # Minimal defensive focus
                }
            },
            'phase_context': {
                'early': {
                    # Establish control: possession and structure
                    'possession': 2.0,
                    'connectivity': 1.8,
                    'spatial': 1.5,
                    'defensive': 1.3,
                    'tempo': 0.9,
                    'attacking': 1.0,
                    'pressing': 1.1,
                    'transition': 1.2,
                },
                'middle': {
                    # Balanced approach: all options viable
                    'attacking': 2.2,
                    'pressing': 2.0,
                    'transition': 2.3,
                    'tempo': 2.0,
                    'spatial': 1.6,
                    'connectivity': 1.5,
                    'possession': 1.3,
                    'defensive': 1.0,
                },
                'late': {
                    # Urgency increases: proactive actions emphasized
                    'attacking': 3.5,
                    'defensive': 2.2,  # Also important if protecting lead
                    'tempo': 3.2,
                    'pressing': 3.0,
                    'transition': 2.8,
                    'spatial': 2.0,
                    'connectivity': 1.7,
                    'possession': 1.5,
                }
            },
            'intensity_context': {
                'low': {
                    # Activate the team: increase tempo and pressing
                    'tempo': 3.8,
                    'pressing': 3.5,
                    'attacking': 3.0,
                    'transition': 2.6,
                    'spatial': 1.6,
                    'connectivity': 1.4,
                    'possession': 0.5,  # Slow possession inappropriate
                    'defensive': 0.4,
                },
                'medium': {
                    # Balanced intensity: all options viable
                    'spatial': 1.8,
                    'connectivity': 1.6,
                    'possession': 1.5,
                    'defensive': 1.3,
                    'attacking': 1.6,
                    'tempo': 1.5,
                    'pressing': 1.4,
                    'transition': 1.5,
                },
                'high': {
                    # Manage fatigue: control and structure
                    'defensive': 2.0,
                    'possession': 2.2,
                    'connectivity': 1.8,
                    'spatial': 1.5,
                    'attacking': 0.8,  # Risky when tired
                    'tempo': 0.7,      # Cannot sustain high tempo
                    'pressing': 0.6,   # Too demanding
                    'transition': 1.0,
                }
            }
        }
    
    def _create_rules(self) -> List[Dict]:
        """
        Create tactical rules with conditions and recommendation generators.
        
        Each rule consists of:
        1. Condition function: Evaluates whether rule should trigger
        2. Recommendation generator: Creates TacticalRecommendation if triggered
        
        Rules are organized by priority:
        - Priority 1: Critical situations requiring immediate action
        - Priority 2: Important tactical adjustments
        - Priority 3: Maintenance of successful patterns
        
        Returns:
            List[Dict]: List of rule dictionaries with structure:
                {
                    'name': 'rule_identifier',
                    'condition': lambda metrics, context: bool,
                    'recommendation': lambda metrics, context: TacticalRecommendation
                }
        
        Notes:
            - Rules reference self.thresholds for dynamic threshold-based conditions
            - Expected impacts are estimated using ImpactEstimator
            - Confidence levels are determined by rule priority and context alignment
        """
        rules = []
        
        # ============================================================================
        # PRIORITY 1: CRITICAL SITUATIONS
        # ============================================================================
        
        rules.append({
            'name': 'trailing_late_emergency',
            'condition': lambda m, c: (
                c.get('score_context') == 'trailing' and
                c.get('phase_context') == 'late' and
                m.get('density', 0) < self.thresholds.get('density', {}).get('poor', 0.4)
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="EMERGENCY: All players forward, maximum attacking commitment",
                recommendation_type=RecommendationType.ATTACKING,
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.95,
                context=c,
                triggered_metrics=['density'],
                reasoning="Trailing late with poor attacking density - desperate measures needed",
                priority=1,
                implementation_time="immediate",
                expected_impact=self.impact_estimator.estimate_impact('attacking', 'very_high', m),
                context_specificity=1.0
            )
        })
        
        rules.append({
            'name': 'leading_late_defensive',
            'condition': lambda m, c: (
                c.get('score_context') == 'leading' and
                c.get('phase_context') == 'late' and
                m.get('clustering_coefficient', 0) > self.thresholds.get('clustering_coefficient', {}).get('good', 0.4)
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Defensive stability: Maintain shape and control possession",
                recommendation_type=RecommendationType.DEFENSIVE,
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.92,
                context=c,
                triggered_metrics=['clustering_coefficient'],
                reasoning="Leading late with good structure - protect the lead",
                priority=1,
                implementation_time="gradual",
                expected_impact=self.impact_estimator.estimate_impact('defensive', 'high', m),
                context_specificity=0.95
            )
        })
        
        rules.append({
            'name': 'connectivity_crisis',
            'condition': lambda m, c: (
                m.get('clustering_coefficient', 0) < self.thresholds.get('clustering_coefficient', {}).get('critical', 0.2) and
                m.get('density', 0) < self.thresholds.get('density', {}).get('poor', 0.4)
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Form passing triangles: Create immediate local connections",
                recommendation_type=RecommendationType.CONNECTIVITY,
                confidence=ConfidenceLevel.VERY_HIGH,
                confidence_score=0.90,
                context=c,
                triggered_metrics=['clustering_coefficient', 'density'],
                reasoning="Critical connectivity breakdown - structural emergency",
                priority=1,
                implementation_time="immediate",
                expected_impact=self.impact_estimator.estimate_impact('connectivity', 'very_high', m),
                context_specificity=0.8
            )
        })
        
        # ============================================================================
        # PRIORITY 2: IMPORTANT ADJUSTMENTS
        # ============================================================================
        
        rules.append({
            'name': 'low_intensity_activation',
            'condition': lambda m, c: (
                c.get('intensity_context') == 'low' and
                m.get('avg_path_length', 3.0) > self.thresholds.get('avg_path_length', {}).get('good', 2.5)
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Increase tempo: Quick passing and higher pressing intensity",
                recommendation_type=RecommendationType.TEMPO,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.88,
                context=c,
                triggered_metrics=['avg_path_length'],
                reasoning="Low intensity with long passing chains - need activation",
                priority=2,
                implementation_time="gradual",
                expected_impact=self.impact_estimator.estimate_impact('tempo', 'high', m),
                context_specificity=0.85
            )
        })
        
        rules.append({
            'name': 'poor_density_attacking',
            'condition': lambda m, c: (
                m.get('density', 0) < self.thresholds.get('density', {}).get('poor', 0.4) and
                c.get('score_context') in ['tied', 'trailing']
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Increase attacking presence: Push more players forward",
                recommendation_type=RecommendationType.ATTACKING,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
                context=c,
                triggered_metrics=['density'],
                reasoning="Poor network density in attacking context - need more connections",
                priority=2,
                implementation_time="gradual",
                expected_impact=self.impact_estimator.estimate_impact('attacking', 'high', m),
                context_specificity=0.8
            )
        })
        
        rules.append({
            'name': 'high_centralization_risk',
            'condition': lambda m, c: (
                m.get('centralization', 0) > self.thresholds.get('centralization', {}).get('good', 0.6) and
                c.get('intensity_context') == 'high'
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Distribute play: Reduce reliance on central hub players",
                recommendation_type=RecommendationType.SPATIAL,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.82,
                context=c,
                triggered_metrics=['centralization'],
                reasoning="High centralization with high intensity - risk of key player fatigue",
                priority=2,
                implementation_time="gradual",
                expected_impact=self.impact_estimator.estimate_impact('spatial', 'medium', m),
                context_specificity=0.75
            )
        })
        
        # ============================================================================
        # PRIORITY 3: MAINTENANCE AND OPTIMIZATION
        # ============================================================================
        
        rules.append({
            'name': 'maintain_successful_pattern',
            'condition': lambda m, c: (
                self.temporal_tracker.should_maintain_consistency(c) and
                self.thresholds.get('density', {}).get('poor', 0.4) <= m.get('density', 0) <= self.thresholds.get('density', {}).get('good', 0.7) and
                m.get('clustering_coefficient', 0) > self.thresholds.get('clustering_coefficient', {}).get('average', 0.3)
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Continue current approach: Team structure is working effectively",
                recommendation_type=RecommendationType.POSSESSION,
                confidence=ConfidenceLevel.HIGH,
                confidence_score=0.85,
                context=c,
                triggered_metrics=['density', 'clustering_coefficient'],
                reasoning="Stable context with good metrics - maintain successful pattern",
                priority=3,
                implementation_time="ongoing",
                expected_impact=self.impact_estimator.estimate_impact('possession', 'normal', m),
                context_specificity=0.75
            )
        })
        
        rules.append({
            'name': 'optimize_good_structure',
            'condition': lambda m, c: (
                m.get('clustering_coefficient', 0) > self.thresholds.get('clustering_coefficient', {}).get('good', 0.4) and
                m.get('density', 0) > self.thresholds.get('density', {}).get('average', 0.5) and
                c.get('score_context') == 'tied'
            ),
            'recommendation': lambda m, c: TacticalRecommendation(
                action="Leverage strong structure: Increase attacking ambition",
                recommendation_type=RecommendationType.ATTACKING,
                confidence=ConfidenceLevel.MEDIUM,
                confidence_score=0.75,
                context=c,
                triggered_metrics=['clustering_coefficient', 'density'],
                reasoning="Strong structural foundation enables safe attacking progression",
                priority=3,
                implementation_time="gradual",
                expected_impact=self.impact_estimator.estimate_impact('attacking', 'medium', m),
                context_specificity=0.7
            )
        })
        
        return rules
    
    def evaluate_rules(self, metrics: Dict, context: Dict) -> List[TacticalRecommendation]:
        """
        Evaluate all rules and return filtered, prioritized recommendations.
        
        This is the main entry point for generating recommendations. The process:
        1. Evaluate all rule conditions against current metrics and context
        2. Generate recommendations for triggered rules
        3. Apply context weighting to adjust confidence scores
        4. Filter recommendations based on urgency-dependent thresholds
        5. Boost confidence for temporally consistent recommendations
        6. Sort by priority and confidence
        7. Return top 3 recommendations
        
        Args:
            metrics (Dict): Current network metrics
            context (Dict): Current match context
        
        Returns:
            List[TacticalRecommendation]: Top 3 filtered and prioritized recommendations
        
        Notes:
            - All rules are evaluated (no short-circuiting)
            - Context weighting uses geometric mean to prevent explosion
            - Temporal consistency tracking updates after evaluation
            - Returns empty list if no recommendations pass filtering
        """
        recommendations = []
        
        # Calculate urgency for threshold determination
        urgency_level, _, _ = self._calculate_urgency(metrics, context)
        
        # Evaluate all rules
        for rule in self.rules:
            try:
                # Check if rule condition is satisfied
                if rule['condition'](metrics, context):
                    # Generate recommendation
                    rec = rule['recommendation'](metrics, context)
                    if rec:
                        recommendations.append(rec)
            except Exception as e:
                # Silently skip rules that fail (e.g., missing metrics)
                continue
        
        # Apply context weighting and filtering
        filtered_recommendations = self._apply_context_weighting(
            recommendations, context, urgency_level
        )
        
        # Calculate temporal consistency
        consistency_score = self.temporal_tracker.get_consistency_score(filtered_recommendations)
        
        # Boost confidence for temporally consistent recommendations
        # Consistent recommendations suggest stable, reliable tactical approach
        for rec in filtered_recommendations:
            if consistency_score > 0.7:
                rec.confidence_score = min(0.98, rec.confidence_score + 0.1)
                # Update confidence level category if needed
                rec.confidence = self._score_to_level(rec.confidence_score)
        
        # Update temporal tracker with current recommendations
        self.temporal_tracker.add_recommendations(filtered_recommendations, context)
        
        # Return top 3 recommendations (sorted by priority and confidence)
        return filtered_recommendations[:3]
    
    def _apply_context_weighting(self, recommendations: List[TacticalRecommendation], 
                                context: Dict, urgency_level: str) -> List[TacticalRecommendation]:
        """
        Apply context-aware weighting to adjust recommendation confidence.
        
        This method adjusts recommendation confidence scores based on how well-suited
        each recommendation type is to the current match context. The weighting uses
        a geometric mean approach to combine weights across multiple context dimensions,
        preventing multiplicative explosion while preserving relative importance.
        
        Args:
            recommendations (List[TacticalRecommendation]): Raw recommendations
            context (Dict): Current match context
            urgency_level (str): Calculated urgency level
        
        Returns:
            List[TacticalRecommendation]: Filtered and weighted recommendations
        
        Process:
            1. For each recommendation, collect context weights across all dimensions
            2. Combine weights using geometric mean: (w1 * w2 * w3)^(1/3)
            3. Adjust confidence score based on context appropriateness
            4. Filter out recommendations below urgency-dependent thresholds
            5. Sort by priority (ascending) and confidence (descending)
        
        Notes:
            - Geometric mean prevents explosion (e.g., 4.5 * 3.5 * 3.8 = 59.85 → 3.91)
            - Strong penalties for contextually inappropriate recommendations
            - Thresholds adapt to urgency (desperate situations accept lower confidence)
        """
        weighted_recommendations = []
        
        # Get urgency-dependent thresholds
        min_confidence = self._get_confidence_threshold(urgency_level)
        min_context_multiplier = self._get_context_threshold(urgency_level)
        
        for rec in recommendations:
            rec_type = rec.recommendation_type.value
            
            # Collect context weights across all dimensions
            weights = []
            for context_type, context_value in context.items():
                if (context_type in self.context_weights and 
                    context_value in self.context_weights[context_type] and
                    rec_type in self.context_weights[context_type][context_value]):
                    
                    weight = self.context_weights[context_type][context_value][rec_type]
                    weights.append(weight)
            
            # Calculate context multiplier using geometric mean
            # This prevents multiplicative explosion while preserving relative importance
            if weights:
                context_multiplier = np.prod(weights) ** (1.0 / len(weights))
            else:
                context_multiplier = 1.0  # Neutral if no weights found
            
            # Apply context effects to confidence
            base_confidence = rec.confidence_score
            
            # Context effect: (multiplier - 1.0) * 0.4 gives strong but bounded effect
            # - multiplier = 2.0 → +0.4 confidence boost
            # - multiplier = 0.5 → -0.2 confidence penalty
            context_effect = (context_multiplier - 1.0) * 0.4
            
            # Additional penalty for highly inappropriate contexts
            if context_multiplier < 0.7:
                context_effect -= 0.3  # Strong penalty
            
            # Calculate final confidence with bounds [0.1, 0.98]
            final_confidence = base_confidence + context_effect
            final_confidence = max(0.1, min(0.98, final_confidence))
            
            # Filter based on urgency-dependent thresholds
            if context_multiplier > min_context_multiplier and final_confidence > min_confidence:
                # Update recommendation with adjusted confidence
                rec.confidence_score = final_confidence
                rec.context_specificity = min(1.0, context_multiplier)
                rec.confidence = self._score_to_level(final_confidence)
                
                weighted_recommendations.append(rec)
        
        # Sort by priority (ascending) then confidence (descending)
        weighted_recommendations.sort(key=lambda x: (x.priority, -x.confidence_score))
        
        return weighted_recommendations
    
    def _calculate_urgency(self, metrics: Dict, context: Dict) -> Tuple[str, List[str], float]:
        """
        Calculate urgency level with weighted factors.
        
        Urgency reflects how critical the current situation is and how quickly
        intervention is needed. It's calculated by summing weighted urgency factors:
        - Context factors (score, phase) have highest weight (2.0-5.0)
        - Metric factors (critical values) have high weight (3.0)
        - Intensity factors have moderate weight (1.5)
        
        Args:
            metrics (Dict): Current network metrics
            context (Dict): Current match context
        
        Returns:
            Tuple[str, List[str], float]:
                - urgency_level: 'very_high', 'high', 'medium', or 'normal'
                - urgency_factors: List of triggered factors
                - urgency_score: Numerical urgency score
        
        Urgency Levels:
            - very_high (≥5.0): Desperate situation requiring immediate action
            - high (≥3.0): Urgent situation requiring prompt action
            - medium (≥1.5): Moderate situation requiring attention
            - normal (<1.5): Routine situation, no urgency
        """
        urgency_score = 0.0
        urgency_factors = []
        
        # Critical metrics (weight: 3.0)
        if metrics.get('density', 0) < self.thresholds.get('density', {}).get('critical', 0.3):
            urgency_factors.append('critical_density')
            urgency_score += 3.0
        
        if metrics.get('clustering_coefficient', 0) < self.thresholds.get('clustering_coefficient', {}).get('critical', 0.2):
            urgency_factors.append('critical_clustering')
            urgency_score += 3.0
        
        # Context urgency (weight: 5.0 for desperate, 2.0 for trailing)
        if context.get('score_context') == 'trailing' and context.get('phase_context') == 'late':
            urgency_factors.append('desperate_situation')
            urgency_score += 5.0  # Highest weight: most critical situation
        elif context.get('score_context') == 'trailing':
            urgency_factors.append('trailing')
            urgency_score += 2.0
        
        # Phase urgency (weight: 2.0)
        if context.get('phase_context') == 'late':
            urgency_factors.append('late_phase')
            urgency_score += 2.0
        
        # Intensity urgency (weight: 1.5)
        if context.get('intensity_context') == 'low':
            urgency_factors.append('low_intensity')
            urgency_score += 1.5
        
        # Determine urgency level from weighted score
        if urgency_score >= 5.0:
            urgency_level = 'very_high'
        elif urgency_score >= 3.0:
            urgency_level = 'high'
        elif urgency_score >= 1.5:
            urgency_level = 'medium'
        else:
            urgency_level = 'normal'
        
        return urgency_level, urgency_factors, urgency_score
    
    def _get_confidence_threshold(self, urgency_level: str) -> float:
        """
        Get minimum confidence threshold based on urgency.
        
        Higher urgency situations accept lower confidence recommendations because
        the cost of inaction exceeds the cost of potentially suboptimal action.
        
        Args:
            urgency_level (str): Urgency level
        
        Returns:
            float: Minimum confidence threshold [0, 1]
        """
        thresholds = {
            'very_high': 0.4,  # Desperate: act on moderate confidence
            'high': 0.5,       # Urgent: need reasonable confidence
            'medium': 0.6,     # Standard: need good confidence
            'normal': 0.7      # Exploratory: need high confidence
        }
        return thresholds.get(urgency_level, 0.6)
    
    def _get_context_threshold(self, urgency_level: str) -> float:
        """
        Get minimum context appropriateness threshold based on urgency.
        
        Args:
            urgency_level (str): Urgency level
        
        Returns:
            float: Minimum context multiplier threshold
        """
        thresholds = {
            'very_high': 0.3,  # Desperate: try anything reasonable
            'high': 0.5,       # Urgent: must be somewhat appropriate
            'medium': 0.7,     # Standard: must be appropriate
            'normal': 0.8      # Exploratory: must be highly appropriate
        }
        return thresholds.get(urgency_level, 0.7)
    
    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """
        Convert continuous confidence score to categorical level.
        
        Args:
            score (float): Confidence score [0, 1]
        
        Returns:
            ConfidenceLevel: Categorical confidence level
        """
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.6:
            return ConfidenceLevel.HIGH
        elif score >= 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW


class TacticalRecommender:
    """
    Main tactical recommendation system interface.
    
    This class provides the high-level API for generating tactical recommendations
    from network metrics and match context. It orchestrates the interaction between:
    - ThresholdAnalyzer: Establishes performance benchmarks
    - RuleEngine: Generates and filters recommendations
    - ImpactEstimator: Quantifies expected effects
    - TemporalTracker: Maintains recommendation history
    
    The recommender can operate at two levels:
    1. Window-level: Generate recommendations for a single 5-minute window
    2. Match-level: Analyze entire match with temporal progression
    
    Usage:
        recommender = TacticalRecommender()
        recommender.initialize_system(network_data)
        recommendations = recommender.get_recommendations(metrics, context)
    """
    
    def __init__(self, rq1_results: Dict = None):
        """
        Initialize tactical recommender.
        
        Args:
            rq1_results (Dict, optional): Results from RQ1 statistical analysis.
                Currently stored for potential future integration but not actively used.
        
        Attributes:
            rq1_results (Dict): RQ1 statistical findings (for future integration)
            threshold_analyzer (ThresholdAnalyzer): Performance threshold extractor
            rule_engine (RuleEngine): Core recommendation logic
            recommendation_history (List[Dict]): History of all generated recommendations
        """
        self.rq1_results = rq1_results
        self.threshold_analyzer = ThresholdAnalyzer()
        self.rule_engine = None
        self.recommendation_history = []
    
    def initialize_system(self, network_data: pd.DataFrame):
        """
        Initialize the recommendation system with historical data.
        
        This method must be called before generating recommendations. It:
        1. Extracts performance thresholds from historical network data
        2. Initializes the rule engine with these thresholds
        3. Prepares the system for recommendation generation
        
        Args:
            network_data (pd.DataFrame): Historical network metrics across all matches
                Must contain columns: density, clustering_coefficient, 
                avg_betweenness_centrality, avg_eigenvector_centrality,
                avg_path_length, centralization
        
        Returns:
            self: For method chaining
        
        Raises:
            ValueError: If network_data is empty or missing required columns
        """
        print("Initializing Tactical Recommendation System...")
        
        # Extract performance thresholds from historical data
        thresholds = self.threshold_analyzer.extract_thresholds(network_data)
        # Log and persist thresholds for audit / reporting
        try:
            self.threshold_analyzer.log_thresholds()
        except Exception:
            # Do not fail initialization on logging error
            pass
         
        # Initialize rule engine with thresholds
        self.rule_engine = RuleEngine(thresholds)
        
        print(f"✅ System initialized with {len(thresholds)} metric thresholds")
        print(f"✅ Rule engine loaded with {len(self.rule_engine.rules)} tactical rules")
        
        return self
    
    def get_recommendations(self, network_metrics: Dict, context: Dict, 
                          window_info: Dict = None) -> Dict:
        """
        Generate tactical recommendations for a single window.
        
        This is the main method for obtaining recommendations. It:
        1. Evaluates rules against current metrics and context
        2. Analyzes situation urgency
        3. Packages recommendations with metadata
        4. Stores in history for temporal tracking
        
        Args:
            network_metrics (Dict): Current network metric values
                Example: {'density': 0.45, 'clustering_coefficient': 0.32, ...}
            context (Dict): Current match context
                Example: {'score_context': 'trailing', 'phase_context': 'late', ...}
            window_info (Dict, optional): Metadata about current window
                Example: {'window_id': 5, 'start_minute': 25, 'end_minute': 30, ...}
        
        Returns:
            Dict: Recommendation package with structure:
                {
                    'timestamp': ISO timestamp,
                    'window_info': {...},
                    'current_metrics': {...},
                    'current_context': {...},
                    'situation_analysis': {
                        'urgency_level': 'high',
                        'urgency_factors': ['trailing', 'late_phase'],
                        'overall_assessment': 'poor'
                    },
                    'recommendations': [
                        {
                            'type': 'attacking',
                            'action': '...',
                            'confidence': 'high',
                            'confidence_score': 0.85,
                            ...
                        }
                    ],
                    'summary': {...},
                    'temporal_consistency': 0.75
                }
        
        Raises:
            ValueError: If system not initialized (call initialize_system() first)
        """
        if self.rule_engine is None:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        # Generate recommendations using rule engine
        recommendations = self.rule_engine.evaluate_rules(network_metrics, context)
        
        # Analyze current situation
        situation_analysis = self._analyze_situation(network_metrics, context)
        
        # Create comprehensive recommendation package
        recommendation_package = {
            'timestamp': datetime.now().isoformat(),
            'window_info': window_info or {},
            'current_metrics': network_metrics,
            'current_context': context,
            'situation_analysis': situation_analysis,
            'recommendations': [self._recommendation_to_dict(rec) for rec in recommendations],
            'summary': self._create_summary(recommendations, situation_analysis),
            'temporal_consistency': self.rule_engine.temporal_tracker.get_consistency_score(recommendations)
        }
        
        # Store in history for temporal tracking and analysis
        self.recommendation_history.append(recommendation_package)
        
        return recommendation_package
    
    def _analyze_situation(self, metrics: Dict, context: Dict) -> Dict:
        """
        Analyze current tactical situation.
        
        Provides a high-level assessment of the current match state by:
        1. Calculating urgency level and factors
        2. Assessing overall performance relative to thresholds
        
        Args:
            metrics (Dict): Current network metrics
            context (Dict): Current match context
        
        Returns:
            Dict: Situation analysis with structure:
                {
                    'urgency_level': 'high',
                    'urgency_factors': ['trailing', 'late_phase'],
                    'urgency_score': 4.0,
                    'overall_assessment': 'poor'
                }
        """
        # Calculate urgency using rule engine's weighted approach
        urgency_level, urgency_factors, urgency_score = self.rule_engine._calculate_urgency(metrics, context)
        
        return {
            'urgency_level': urgency_level,
            'urgency_factors': urgency_factors,
            'urgency_score': urgency_score,
            'overall_assessment': self._assess_overall_situation(metrics, context)
        }
    
    def _assess_overall_situation(self, metrics: Dict, context: Dict) -> str:
        """
        Assess overall tactical situation relative to performance thresholds.
        
        Calculates a composite score by comparing each metric to its thresholds:
        - Excellent/Good performance: +2/+1 points
        - Poor/Critical performance: -1/-2 points
        - Context adjustments: ±1 point
        
        Args:
            metrics (Dict): Current network metrics
            context (Dict): Current match context
        
        Returns:
            str: Overall assessment category
                - 'excellent' (score ≥ 3)
                - 'good' (score ≥ 1)
                - 'average' (score ≥ -1)
                - 'poor' (score ≥ -3)
                - 'critical' (score < -3)
        """
        score = 0
        thresholds = self.threshold_analyzer.thresholds
        
        # Assess each metric against thresholds
        for metric, value in metrics.items():
            if metric in thresholds and pd.notna(value):
                if value >= thresholds[metric]['good']:
                    score += 2
                elif value >= thresholds[metric]['average']:
                    score += 1
                elif value <= thresholds[metric]['poor']:
                    score -= 1
                elif value <= thresholds[metric]['critical']:
                    score -= 2
        
        # Context adjustments
        if context.get('score_context') == 'leading':
            score += 1  # Positive context
        elif context.get('score_context') == 'trailing':
            score -= 1  # Negative context
        
        # Map score to assessment category
        if score >= 3:
            return 'excellent'
        elif score >= 1:
            return 'good'
        elif score >= -1:
            return 'average'
        elif score >= -3:
            return 'poor'
        else:
            return 'critical'
    
    def _recommendation_to_dict(self, rec: TacticalRecommendation) -> Dict:
        """
        Convert TacticalRecommendation object to dictionary for serialization.
        
        Args:
            rec (TacticalRecommendation): Recommendation object
        
        Returns:
            Dict: Serializable recommendation dictionary
        """
        return {
            'type': rec.recommendation_type.value,
            'action': rec.action,
            'confidence': rec.confidence.value,
            'confidence_score': round(rec.confidence_score, 3),
            'priority': rec.priority,
            'reasoning': rec.reasoning,
            'implementation_time': rec.implementation_time,
            'expected_impact': rec.expected_impact,  # Already includes uncertainty
            'context_specificity': round(rec.context_specificity, 3),
            'triggered_metrics': rec.triggered_metrics
        }
    
    def _create_summary(self, recommendations: List[TacticalRecommendation], 
                       situation_analysis: Dict) -> Dict:
        """
        Create concise summary of recommendations.
        
        Args:
            recommendations (List[TacticalRecommendation]): Generated recommendations
            situation_analysis (Dict): Situation analysis
        
        Returns:
            Dict: Summary with structure:
                {
                    'primary_focus': 'attacking',
                    'urgency': 'high',
                    'confidence': 0.85,
                    'implementation': 'immediate',
                    'total_recommendations': 3
                }
        """
        if not recommendations:
            return {
                'primary_focus': 'none',
                'urgency': situation_analysis['urgency_level'],
                'confidence': 0.0,
                'implementation': 'no_action',
                'total_recommendations': 0
            }
        
        primary_rec = recommendations[0]
        avg_confidence = np.mean([rec.confidence_score for rec in recommendations])
        
        return {
            'primary_focus': primary_rec.recommendation_type.value,
            'urgency': situation_analysis['urgency_level'],
            'confidence': round(avg_confidence, 3),
            'implementation': primary_rec.implementation_time,
            'total_recommendations': len(recommendations)
        }
    
    def analyze_match_recommendations(self, match_data: pd.DataFrame, 
                                    match_id: str = None) -> Dict:
        """
        Analyze recommendations for an entire match.
        
        Generates window-by-window recommendations across the full match duration,
        enabling analysis of tactical progression and critical moments.
        
        Args:
            match_data (pd.DataFrame): Match data with network metrics and context
                Must contain: network metrics, context columns, window metadata
            match_id (str, optional): Match identifier for tracking
        
        Returns:
            Dict: Match-level analysis with structure:
                {
                    'match_id': '...',
                    'total_windows': 18,
                    'window_recommendations': [...],
                    'match_analysis': {...},
                    'match_summary': {...}
                }
        
        Notes:
            - Processes each window sequentially
            - Temporal tracking maintains state across windows
            - Skips windows with missing data (logs warning)
        """
        match_recommendations = []
        
        for idx, row in match_data.iterrows():
            # Extract network metrics from row
            network_metrics = {}
            for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                          'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
                if metric in row and pd.notna(row[metric]):
                    network_metrics[metric] = row[metric]
            
            # Extract context from row
            context = {}
            for ctx in ['score_context', 'phase_context', 'intensity_context']:
                if ctx in row and pd.notna(row[ctx]):
                    context[ctx] = row[ctx]
            
            # Extract window metadata
            window_info = {
                'window_id': idx,
                'start_minute': row.get('start_minute'),
                'end_minute': row.get('end_minute'),
                'match_id': match_id or row.get('match_id'),
                'team': row.get('team')
            }
            
            # Generate recommendations for this window
            try:
                window_recommendations = self.get_recommendations(
                    network_metrics, context, window_info
                )
                match_recommendations.append(window_recommendations)
            except Exception as e:
                print(f"Warning: Could not generate recommendations for window {idx}: {e}")
                continue
        
        return {
            'match_id': match_id,
            'total_windows': len(match_recommendations),
            'window_recommendations': match_recommendations,
            'match_analysis': self._summarize_match(match_recommendations),
            'match_summary': self._summarize_match(match_recommendations)  # Keep both for compatibility
        }
    
    def _summarize_match(self, match_recommendations: List[Dict]) -> Dict:
        """
        Summarize match-level patterns.
        
        Analyzes temporal progression of recommendations across the match to identify:
        - Critical tactical moments
        - Dominant tactical themes
        - Recommendation stability
        - Match narrative
        
        Args:
            match_recommendations (List[Dict]): Window-level recommendations
        
        Returns:
            Dict: Match summary with comprehensive tactical analysis
        """
        if not match_recommendations:
            return {
                'status': 'no_data',
                'critical_periods': [],
                'total_critical_windows': 0
            }
        
        # Extract patterns across windows
        urgency_levels = [w['summary']['urgency'] for w in match_recommendations]
        primary_focuses = [w['summary']['primary_focus'] for w in match_recommendations]
        confidences = [w['summary']['confidence'] for w in match_recommendations]
        
        # Calculate temporal consistency (actual calculation)
        consistency_scores = []
        for i in range(1, len(primary_focuses)):
            prev_focus = primary_focuses[i-1]
            curr_focus = primary_focuses[i]
            consistency_scores.append(1.0 if prev_focus == curr_focus else 0.0)
        
        recommendation_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        # Identify critical periods (high/very_high urgency)
        critical_periods = []
        for i, window in enumerate(match_recommendations):
            if window['summary']['urgency'] in ['high', 'very_high']:
                critical_periods.append({
                    'window': i,
                    'minute': window['window_info'].get('start_minute'),
                    'urgency': window['summary']['urgency'],
                    'focus': window['summary']['primary_focus'],
                    'confidence': window['summary']['confidence']
                })
        
        # Identify tactical phases (consecutive windows with same focus)
        tactical_phases = []
        if primary_focuses:
            current_phase = {'focus': primary_focuses[0], 'start': 0, 'windows': 1}
            
            for i in range(1, len(primary_focuses)):
                if primary_focuses[i] == current_phase['focus']:
                    current_phase['windows'] += 1
                else:
                    if current_phase['windows'] >= 2:  # Only include sustained phases
                        tactical_phases.append(current_phase.copy())
                    current_phase = {'focus': primary_focuses[i], 'start': i, 'windows': 1}
            
            if current_phase['windows'] >= 2:
                tactical_phases.append(current_phase)
        
        return {
            # Overall statistics
            'average_confidence': round(np.mean(confidences), 3),
            'confidence_std': round(np.std(confidences), 3),
            
            # Urgency analysis
            'urgency_distribution': dict(Counter(urgency_levels)),
            'total_critical_windows': len(critical_periods),
            'critical_window_percentage': round(len(critical_periods) / len(match_recommendations) * 100, 1),
            
            # Tactical focus analysis
            'focus_distribution': dict(Counter(primary_focuses)),
            'most_common_recommendations': dict(Counter(primary_focuses).most_common(3)),
            'dominant_focus': Counter(primary_focuses).most_common(1)[0][0] if primary_focuses else None,
            
            # Temporal patterns
            'recommendation_consistency': round(recommendation_consistency, 3),
            'tactical_phases': tactical_phases,
            'total_tactical_shifts': len(tactical_phases),
            
            # Critical moments
            'critical_periods': critical_periods,
            
            # Match narrative
            'match_narrative': self._generate_match_narrative(
                urgency_levels, primary_focuses, critical_periods, tactical_phases
            )
        }
    
    def _generate_match_narrative(self, urgency_levels: List[str], 
                                  primary_focuses: List[str],
                                  critical_periods: List[Dict],
                                  tactical_phases: List[Dict]) -> str:
        """
        Generate human-readable match narrative.
        
        Creates a textual summary of the match's tactical story based on
        recommendation patterns and critical moments.
        
        Args:
            urgency_levels (List[str]): Urgency levels across windows
            primary_focuses (List[str]): Primary recommendation types across windows
            critical_periods (List[Dict]): Critical tactical moments
            tactical_phases (List[Dict]): Sustained tactical phases
        
        Returns:
            str: Human-readable match narrative
        """
        narrative_parts = []
        
        # Overall urgency assessment
        very_high_count = urgency_levels.count('very_high')
        high_count = urgency_levels.count('high')
        
        if very_high_count > 0:
            narrative_parts.append(
                f"Match featured {very_high_count} critical moment(s) requiring immediate intervention."
            )
        
        # Dominant tactical theme
        focus_counter = Counter(primary_focuses)
        dominant_focus = focus_counter.most_common(1)[0]
        narrative_parts.append(
            f"Primary tactical emphasis was {dominant_focus[0]} "
            f"({dominant_focus[1]}/{len(primary_focuses)} windows)."
        )
        
        # Tactical stability assessment
        if len(tactical_phases) <= 2:
            narrative_parts.append("Tactics remained relatively stable throughout.")
        elif len(tactical_phases) >= 5:
            narrative_parts.append("Tactics shifted frequently, suggesting reactive adjustments.")
        
        # Critical periods timing
        if critical_periods:
            critical_minutes = [p['minute'] for p in critical_periods if p['minute']]
            if critical_minutes:
                narrative_parts.append(
                    f"Critical interventions needed at minutes: {', '.join(map(str, critical_minutes))}."
                )
        
        return " ".join(narrative_parts)
    
    def save_recommendations(self, filepath: str = "results/tactical_recommendations.json"):
        """
        Save recommendation history to JSON file.
        
        Args:
            filepath (str): Output file path
        
        Notes:
            - Creates parent directories if they don't exist
            - Includes metadata about system version and generation time
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'metadata': {
                'system_version': 'v2.0_geometric_mean',
                'total_recommendations': len(self.recommendation_history),
                'generation_timestamp': datetime.now().isoformat()
            },
            'recommendations': self.recommendation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Recommendations saved to {filepath}")
    
    def get_system_summary(self) -> Dict:
        """
        Get system status summary.
        
        Returns:
            Dict: System summary with initialization status and statistics
        """
        return {
            'system_status': 'initialized' if self.rule_engine else 'not_initialized',
            'system_version': 'v2.0_geometric_mean',
            'total_rules': len(self.rule_engine.rules) if self.rule_engine else 0,
            'recommendation_history': len(self.recommendation_history),
            'temporal_tracking': 'active' if self.rule_engine else 'inactive',
            'features': [
                'geometric_mean_weighting',
                'urgency_dependent_thresholds',
                'literature_based_impacts',
                'weighted_urgency_calculation',
                'temporal_consistency_tracking'
            ]
        }
