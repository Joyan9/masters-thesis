import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json

class ConfidenceLevel(Enum):
    VERY_HIGH = "very_high"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class RecommendationType(Enum):
    SPATIAL = "spatial"
    TEMPO = "tempo"
    CONNECTIVITY = "connectivity"
    DEFENSIVE = "defensive"
    ATTACKING = "attacking"

@dataclass
class TacticalRecommendation:
    """Structure for tactical recommendations"""
    action: str
    recommendation_type: RecommendationType
    confidence: ConfidenceLevel
    confidence_score: float
    context: str
    triggered_metrics: List[str]
    reasoning: str
    priority: int
    implementation_time: str = "immediate"

@dataclass
class Rule:
    """Structure for tactical rules"""
    rule_id: str
    name: str
    conditions: Dict[str, Any]
    recommendation: TacticalRecommendation
    weight: float
    source: str  # "statistical", "literature", "expert"

class RuleEngine:
    """Core rule-based tactical recommendation engine"""
    
    def __init__(self, thresholds: Dict = None):
        self.thresholds = thresholds or {}
        self.rules = []
        self.rule_database = {}
        self._initialize_rule_database()
    
    def _initialize_rule_database(self):
        """Initialize the rule database with core tactical rules"""
        
        # Spatial Distribution Rules
        self._add_spatial_rules()
        
        # Connectivity Rules
        self._add_connectivity_rules()
        
        # Tempo Rules
        self._add_tempo_rules()
        
        # Context-Adaptive Rules
        self._add_context_adaptive_rules()
        
        # Intensity-Based Rules (from RQ1 findings)
        self._add_intensity_rules()
    
    def _add_spatial_rules(self):
        """Add spatial distribution rules"""
        
        rules = [
            {
                'rule_id': 'SP001',
                'name': 'Low Central Density - Increase Wing Play',
                'conditions': {
                    'density': {'operator': '<', 'threshold': 'poor'},
                    'centralization': {'operator': '<', 'threshold': 'average'}
                },
                'recommendation': TacticalRecommendation(
                    action="Increase wing play and wide positioning",
                    recommendation_type=RecommendationType.SPATIAL,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    context="Low central connectivity detected",
                    triggered_metrics=['density', 'centralization'],
                    reasoning="Low central density suggests congested middle - utilize flanks",
                    priority=2
                ),
                'weight': 0.8,
                'source': 'statistical'
            },
            {
                'rule_id': 'SP002',
                'name': 'High Centralization - Maintain Central Control',
                'conditions': {
                    'centralization': {'operator': '>', 'threshold': 'good'},
                    'density': {'operator': '>', 'threshold': 'average'}
                },
                'recommendation': TacticalRecommendation(
                    action="Maintain possession through central areas",
                    recommendation_type=RecommendationType.SPATIAL,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    context="Strong central control established",
                    triggered_metrics=['centralization', 'density'],
                    reasoning="High centralization indicates effective central control",
                    priority=1
                ),
                'weight': 0.85,
                'source': 'statistical'
            },
            {
                'rule_id': 'SP003',
                'name': 'Low Density + Trailing - Urgent Wing Attack',
                'conditions': {
                    'density': {'operator': '<', 'threshold': 'poor'},
                    'score_context': {'operator': '==', 'value': 'trailing'}
                },
                'recommendation': TacticalRecommendation(
                    action="Urgent wing attacks and overlapping runs",
                    recommendation_type=RecommendationType.ATTACKING,
                    confidence=ConfidenceLevel.VERY_HIGH,
                    confidence_score=0.9,
                    context="Trailing with poor central connectivity",
                    triggered_metrics=['density'],
                    reasoning="Need to create chances quickly through wide areas",
                    priority=1
                ),
                'weight': 0.9,
                'source': 'literature'
            }
        ]
        
        for rule_data in rules:
            rule = Rule(**rule_data)
            self.rules.append(rule)
            self.rule_database[rule.rule_id] = rule
    
    def _add_connectivity_rules(self):
        """Add connectivity-based rules"""
        
        rules = [
            {
                'rule_id': 'CN001',
                'name': 'Low Clustering - Improve Triangles',
                'conditions': {
                    'clustering_coefficient': {'operator': '<', 'threshold': 'poor'}
                },
                'recommendation': TacticalRecommendation(
                    action="Create more passing triangles and short combinations",
                    recommendation_type=RecommendationType.CONNECTIVITY,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    context="Poor local connectivity",
                    triggered_metrics=['clustering_coefficient'],
                    reasoning="Low clustering indicates isolated players - need better triangulation",
                    priority=2
                ),
                'weight': 0.8,
                'source': 'literature'
            },
            {
                'rule_id': 'CN002',
                'name': 'High Path Length - Create Direct Lanes',
                'conditions': {
                    'avg_path_length': {'operator': '>', 'threshold': 'poor'}
                },
                'recommendation': TacticalRecommendation(
                    action="Create more direct passing lanes and reduce ball circulation",
                    recommendation_type=RecommendationType.CONNECTIVITY,
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.7,
                    context="Inefficient ball circulation",
                    triggered_metrics=['avg_path_length'],
                    reasoning="High path length suggests inefficient passing - need more direct routes",
                    priority=3
                ),
                'weight': 0.7,
                'source': 'statistical'
            },
            {
                'rule_id': 'CN003',
                'name': 'Low Betweenness - Activate Playmakers',
                'conditions': {
                    'avg_betweenness_centrality': {'operator': '<', 'threshold': 'poor'}
                },
                'recommendation': TacticalRecommendation(
                    action="Activate central playmakers and improve ball distribution",
                    recommendation_type=RecommendationType.CONNECTIVITY,
                    confidence=ConfidenceLevel.MEDIUM,
                    confidence_score=0.75,
                    context="Weak central distribution",
                    triggered_metrics=['avg_betweenness_centrality'],
                    reasoning="Low betweenness suggests no clear playmaker - need distribution hub",
                    priority=2
                ),
                'weight': 0.75,
                'source': 'expert'
            }
        ]
        
        for rule_data in rules:
            rule = Rule(**rule_data)
            self.rules.append(rule)
            self.rule_database[rule.rule_id] = rule
    
    def _add_tempo_rules(self):
        """Add tempo-based rules"""
        
        rules = [
            {
                'rule_id': 'TP001',
                'name': 'Low Intensity - Increase Tempo',
                'conditions': {
                    'intensity_context': {'operator': '==', 'value': 'low'},
                    'phase_context': {'operator': '!=', 'value': 'late'}
                },
                'recommendation': TacticalRecommendation(
                    action="Increase passing frequency and movement intensity",
                    recommendation_type=RecommendationType.TEMPO,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    context="Low intensity play detected",
                    triggered_metrics=['intensity_context'],
                    reasoning="Low intensity may indicate passive play - increase tempo",
                    priority=2
                ),
                'weight': 0.8,
                'source': 'statistical'
            },
            {
                'rule_id': 'TP002',
                'name': 'High Intensity + Leading - Control Tempo',
                'conditions': {
                    'intensity_context': {'operator': '==', 'value': 'high'},
                    'score_context': {'operator': '==', 'value': 'leading'}
                },
                'recommendation': TacticalRecommendation(
                    action="Control tempo and maintain possession",
                    recommendation_type=RecommendationType.TEMPO,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    context="Leading with high intensity",
                    triggered_metrics=['intensity_context'],
                    reasoning="When leading, control tempo to manage the game",
                    priority=1
                ),
                'weight': 0.85,
                'source': 'literature'
            }
        ]
        
        for rule_data in rules:
            rule = Rule(**rule_data)
            self.rules.append(rule)
            self.rule_database[rule.rule_id] = rule
    
    def _add_context_adaptive_rules(self):
        """Add context-adaptive rules"""
        
        rules = [
            {
                'rule_id': 'CA001',
                'name': 'Late Game + Trailing + Low Density',
                'conditions': {
                    'phase_context': {'operator': '==', 'value': 'late'},
                    'score_context': {'operator': '==', 'value': 'trailing'},
                    'density': {'operator': '<', 'threshold': 'average'}
                },
                'recommendation': TacticalRecommendation(
                    action="Urgent attacking tempo with direct play and wing focus",
                    recommendation_type=RecommendationType.ATTACKING,
                    confidence=ConfidenceLevel.VERY_HIGH,
                    confidence_score=0.95,
                    context="Critical late game situation",
                    triggered_metrics=['phase_context', 'density'],
                    reasoning="Late game deficit requires urgent attacking changes",
                    priority=1,
                    implementation_time="immediate"
                ),
                'weight': 0.95,
                'source': 'expert'
            },
            {
                'rule_id': 'CA002',
                'name': 'Early Game + Leading + High Centralization',
                'conditions': {
                    'phase_context': {'operator': '==', 'value': 'early'},
                    'score_context': {'operator': '==', 'value': 'leading'},
                    'centralization': {'operator': '>', 'threshold': 'good'}
                },
                'recommendation': TacticalRecommendation(
                    action="Maintain central control and manage game tempo",
                    recommendation_type=RecommendationType.DEFENSIVE,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.8,
                    context="Early lead with good control",
                    triggered_metrics=['centralization'],
                    reasoning="Early lead allows for controlled, patient approach",
                    priority=2
                ),
                'weight': 0.8,
                'source': 'literature'
            }
        ]
        
        for rule_data in rules:
            rule = Rule(**rule_data)
            self.rules.append(rule)
            self.rule_database[rule.rule_id] = rule
    
    def _add_intensity_rules(self):
        """Add intensity-based rules from RQ1 findings"""
        
        rules = [
            {
                'rule_id': 'IN001',
                'name': 'Target High Intensity Network Structure',
                'conditions': {
                    'intensity_context': {'operator': '==', 'value': 'low'},
                    'density': {'operator': '<', 'threshold': 'average'}
                },
                'recommendation': TacticalRecommendation(
                    action="Increase network connectivity to match high-intensity patterns",
                    recommendation_type=RecommendationType.TEMPO,
                    confidence=ConfidenceLevel.VERY_HIGH,
                    confidence_score=0.9,
                    context="RQ1 finding: Intensity drives network structure",
                    triggered_metrics=['intensity_context', 'density'],
                    reasoning="RQ1 shows high intensity creates 117% higher density - target this structure",
                    priority=1
                ),
                'weight': 0.9,
                'source': 'statistical'
            },
            {
                'rule_id': 'IN002',
                'name': 'Leverage Intensity-Density Relationship',
                'conditions': {
                    'density': {'operator': '<', 'threshold': 'poor'}
                },
                'recommendation': TacticalRecommendation(
                    action="Increase match intensity to naturally improve network density",
                    recommendation_type=RecommendationType.TEMPO,
                    confidence=ConfidenceLevel.HIGH,
                    confidence_score=0.85,
                    context="Based on RQ1 large effect size (η² = 0.388)",
                    triggered_metrics=['density'],
                    reasoning="Strong statistical evidence that intensity drives density improvements",
                    priority=1
                ),
                'weight': 0.85,
                'source': 'statistical'
            }
        ]
        
        for rule_data in rules:
            rule = Rule(**rule_data)
            self.rules.append(rule)
            self.rule_database[rule.rule_id] = rule
    
    def evaluate_rules(self, network_metrics: Dict, context: Dict) -> List[TacticalRecommendation]:
        """
        Evaluate all rules against current network metrics and context
        
        Args:
            network_metrics: Current window's network metrics
            context: Current context (score, phase, intensity)
        
        Returns:
            List of applicable tactical recommendations
        """
        
        applicable_recommendations = []
        
        for rule in self.rules:
            if self._rule_matches(rule, network_metrics, context):
                # Calculate dynamic confidence based on threshold deviation
                dynamic_confidence = self._calculate_dynamic_confidence(
                    rule, network_metrics, context
                )
                
                # Create recommendation with updated confidence
                recommendation = rule.recommendation
                recommendation.confidence_score = dynamic_confidence
                recommendation.confidence = self._score_to_confidence_level(dynamic_confidence)
                
                applicable_recommendations.append(recommendation)
        
        # Sort by priority and confidence
        applicable_recommendations.sort(
            key=lambda x: (x.priority, -x.confidence_score)
        )
        
        # Resolve conflicts
        resolved_recommendations = self._resolve_conflicts(applicable_recommendations)
        
        return resolved_recommendations
    
    def _rule_matches(self, rule: Rule, metrics: Dict, context: Dict) -> bool:
        """Check if a rule's conditions are met"""
        
        for condition_key, condition in rule.conditions.items():
            
            # Get the value to check
            if condition_key in metrics:
                current_value = metrics[condition_key]
            elif condition_key in context:
                current_value = context[condition_key]
            else:
                continue  # Skip if metric/context not available
            
            # Check condition
            if not self._evaluate_condition(condition_key, current_value, condition):
                return False
        
        return True
    
    def _evaluate_condition(self, metric_name: str, current_value: float, 
                          condition: Dict) -> bool:
        """Evaluate a single condition"""
        
        operator = condition['operator']
        
        if 'threshold' in condition:
            # Threshold-based condition
            threshold_type = condition['threshold']
            threshold_value = self._get_threshold_value(metric_name, threshold_type)
            
            if threshold_value is None:
                return False
            
            if operator == '<':
                return current_value < threshold_value
            elif operator == '>':
                return current_value > threshold_value
            elif operator == '<=':
                return current_value <= threshold_value
            elif operator == '>=':
                return current_value >= threshold_value
            elif operator == '==':
                return abs(current_value - threshold_value) < 0.001
        
        elif 'value' in condition:
            # Direct value condition
            target_value = condition['value']
            
            if operator == '==':
                return current_value == target_value
            elif operator == '!=':
                return current_value != target_value
        
        return False
    
    def _get_threshold_value(self, metric_name: str, threshold_type: str) -> Optional[float]:
        """Get threshold value for a metric"""
        
        if metric_name not in self.thresholds:
            return None
        
        metric_thresholds = self.thresholds[metric_name]
        
        if threshold_type in metric_thresholds.get('percentiles', {}):
            return metric_thresholds['percentiles'][threshold_type]
        elif threshold_type in metric_thresholds.get('statistical', {}):
            return metric_thresholds['statistical'][threshold_type]
        
        return None
    
    def _calculate_dynamic_confidence(self, rule: Rule, metrics: Dict, 
                                    context: Dict) -> float:
        """Calculate dynamic confidence based on how far metrics deviate from thresholds"""
        
        base_confidence = rule.recommendation.confidence_score
        
        # Calculate deviation factor
        total_deviation = 0
        condition_count = 0
        
        for condition_key, condition in rule.conditions.items():
            if condition_key in metrics and 'threshold' in condition:
                current_value = metrics[condition_key]
                threshold_value = self._get_threshold_value(condition_key, condition['threshold'])
                
                if threshold_value is not None:
                    # Calculate normalized deviation
                    if condition_key in self.thresholds:
                        metric_range = (
                            self.thresholds[condition_key]['range']['max'] - 
                            self.thresholds[condition_key]['range']['min']
                        )
                        if metric_range > 0:
                            deviation = abs(current_value - threshold_value) / metric_range
                            total_deviation += deviation
                            condition_count += 1
        
        if condition_count > 0:
            avg_deviation = total_deviation / condition_count
            # Boost confidence for larger deviations (more extreme situations)
            confidence_boost = min(0.2, avg_deviation * 0.5)
            return min(1.0, base_confidence + confidence_boost)
        
        return base_confidence
    
    def _score_to_confidence_level(self, score: float) -> ConfidenceLevel:
        """Convert confidence score to confidence level"""
        
        if score >= 0.9:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.8:
            return ConfidenceLevel.HIGH
        elif score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _resolve_conflicts(self, recommendations: List[TacticalRecommendation]) -> List[TacticalRecommendation]:
        """Resolve conflicting recommendations"""
        
        # Group by recommendation type
        type_groups = {}
        for rec in recommendations:
            rec_type = rec.recommendation_type
            if rec_type not in type_groups:
                type_groups[rec_type] = []
            type_groups[rec_type].append(rec)
        
        # Keep highest confidence recommendation per type
        resolved = []
        for rec_type, recs in type_groups.items():
            best_rec = max(recs, key=lambda x: x.confidence_score)
            resolved.append(best_rec)
        
        return resolved
    
    def get_rule_summary(self) -> Dict:
        """Get summary of available rules"""
        
        summary = {
            'total_rules': len(self.rules),
            'by_type': {},
            'by_source': {},
            'rule_list': []
        }
        
        for rule in self.rules:
            # Count by type
            rec_type = rule.recommendation.recommendation_type.value
            summary['by_type'][rec_type] = summary['by_type'].get(rec_type, 0) + 1
            
            # Count by source
            summary['by_source'][rule.source] = summary['by_source'].get(rule.source, 0) + 1
            
            # Add to rule list
            summary['rule_list'].append({
                'id': rule.rule_id,
                'name': rule.name,
                'type': rec_type,
                'source': rule.source,
                'weight': rule.weight
            })
        
        return summary
