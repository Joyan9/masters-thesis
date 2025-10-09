import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")


class RecommendationType(Enum):
    SPATIAL = "spatial"
    TEMPO = "tempo" 
    CONNECTIVITY = "connectivity"
    ATTACKING = "attacking"
    DEFENSIVE = "defensive"
    PRESSING = "pressing"
    POSSESSION = "possession"
    TRANSITION = "transition"

class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class TacticalRecommendation:
    action: str
    recommendation_type: RecommendationType
    confidence: ConfidenceLevel
    confidence_score: float
    context: Dict
    triggered_metrics: List[str]
    reasoning: str
    priority: int
    implementation_time: str
    expected_impact: Dict[str, float]
    context_specificity: float

class TemporalTracker:
    """Simple temporal consistency tracking"""
    
    def __init__(self):
        self.recent_recommendations = []
        self.recent_contexts = []
        
    def add_recommendations(self, recommendations: List[TacticalRecommendation], context: Dict):
        """Add recommendations to history"""
        rec_types = [rec.recommendation_type.value for rec in recommendations]
        self.recent_recommendations.append(rec_types)
        self.recent_contexts.append(context.copy())
        
        # Keep only last 5 windows
        if len(self.recent_recommendations) > 5:
            self.recent_recommendations.pop(0)
            self.recent_contexts.pop(0)
    
    def get_consistency_score(self, current_recs: List[TacticalRecommendation]) -> float:
        """Calculate temporal consistency score"""
        if len(self.recent_recommendations) < 2:
            return 0.8
        
        current_types = [rec.recommendation_type.value for rec in current_recs]
        
        # Compare with last 2 windows
        similarities = []
        for past_types in self.recent_recommendations[-2:]:
            if past_types and current_types:
                common = len(set(current_types).intersection(set(past_types)))
                total = len(set(current_types).union(set(past_types)))
                similarity = common / total if total > 0 else 0
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def should_maintain_consistency(self, context: Dict) -> bool:
        """Check if context suggests maintaining consistency"""
        if len(self.recent_contexts) < 2:
            return False
        
        # Check if context has been stable
        last_context = self.recent_contexts[-1]
        stable_factors = 0
        total_factors = 0
        
        for key in ['score_context', 'phase_context', 'intensity_context']:
            if key in context and key in last_context:
                total_factors += 1
                if context[key] == last_context[key]:
                    stable_factors += 1
        
        return stable_factors / total_factors > 0.6 if total_factors > 0 else False

class ThresholdAnalyzer:
    """Simple threshold analysis"""
    
    def __init__(self):
        self.thresholds = {}
    
    def extract_thresholds(self, network_data: pd.DataFrame) -> Dict:
        """Extract performance thresholds"""
        thresholds = {}
        
        metrics = ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                  'avg_eigenvector_centrality', 'avg_path_length', 'centralization']
        
        for metric in metrics:
            if metric in network_data.columns:
                values = network_data[metric].dropna()
                if len(values) > 0:
                    thresholds[metric] = {
                        'excellent': np.percentile(values, 90),
                        'good': np.percentile(values, 75),
                        'average': np.percentile(values, 50),
                        'poor': np.percentile(values, 25),
                        'critical': np.percentile(values, 10)
                    }
        
        self.thresholds = thresholds
        return thresholds

class RuleEngine:
    """Simplified rule engine focused on validation improvements"""
    
    def __init__(self, thresholds: Dict):
        self.thresholds = thresholds
        self.temporal_tracker = TemporalTracker()
        self.context_weights = self._create_context_weights()
        self.rules = self._create_rules()
    
    def _create_context_weights(self) -> Dict:
        """Create strong context differentiation weights"""
        return {
            'score_context': {
                'leading': {
                    'defensive': 3.0,
                    'possession': 2.5,
                    'connectivity': 1.8,
                    'spatial': 0.3,
                    'attacking': 0.2,
                    'tempo': 0.5,
                    'pressing': 0.4,
                    'transition': 0.7
                },
                'tied': {
                    'attacking': 1.8,
                    'tempo': 2.0,
                    'pressing': 1.7,
                    'transition': 1.9,
                    'spatial': 1.5,
                    'connectivity': 1.3,
                    'possession': 1.1,
                    'defensive': 0.8
                },
                'trailing': {
                    'attacking': 3.5,
                    'tempo': 3.0,
                    'pressing': 2.8,
                    'transition': 3.2,
                    'spatial': 2.0,
                    'connectivity': 1.5,
                    'possession': 0.2,
                    'defensive': 0.1
                }
            },
            'phase_context': {
                'early': {
                    'possession': 2.0,
                    'connectivity': 1.8,
                    'spatial': 1.6,
                    'defensive': 1.4,
                    'tempo': 0.7,
                    'attacking': 0.8,
                    'pressing': 0.9,
                    'transition': 1.0
                },
                'middle': {
                    'attacking': 1.6,
                    'pressing': 1.5,
                    'transition': 1.7,
                    'tempo': 1.4,
                    'spatial': 1.3,
                    'connectivity': 1.2,
                    'possession': 1.1,
                    'defensive': 1.0
                },
                'late': {
                    'attacking': 2.5,
                    'defensive': 2.2,
                    'tempo': 2.3,
                    'pressing': 2.0,
                    'transition': 1.8,
                    'spatial': 1.5,
                    'connectivity': 1.3,
                    'possession': 1.6
                }
            },
            'intensity_context': {
                'low': {
                    'tempo': 3.0,
                    'pressing': 2.5,
                    'attacking': 2.0,
                    'transition': 1.8,
                    'spatial': 1.2,
                    'connectivity': 1.1,
                    'possession': 0.8,
                    'defensive': 0.7
                },
                'medium': {
                    'spatial': 1.5,
                    'connectivity': 1.4,
                    'possession': 1.3,
                    'defensive': 1.2,
                    'attacking': 1.1,
                    'tempo': 1.0,
                    'pressing': 1.0,
                    'transition': 1.1
                },
                'high': {
                    'defensive': 2.0,
                    'possession': 2.2,
                    'connectivity': 1.8,
                    'spatial': 1.5,
                    'attacking': 0.6,
                    'tempo': 0.5,
                    'pressing': 0.4,
                    'transition': 1.0
                }
            }
        }
    
    def _create_rules(self) -> List:
        """Create focused tactical rules"""
        rules = []
        
        # Critical situation rules with high impact
        rules.append({
            'name': 'trailing_late_emergency',
            'condition': lambda m, c: (
                c.get('score_context') == 'trailing' and
                c.get('phase_context') == 'late' and
                m.get('density', 0) < 0.5
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
                expected_impact={'density': 0.25, 'centralization': 0.20},
                context_specificity=1.0
            )
        })
        
        rules.append({
            'name': 'leading_late_defensive',
            'condition': lambda m, c: (
                c.get('score_context') == 'leading' and
                c.get('phase_context') == 'late' and
                m.get('clustering_coefficient', 0) > 0.4
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
                expected_impact={'clustering_coefficient': 0.08, 'centralization': -0.05},
                context_specificity=0.95
            )
        })
        
        # Low intensity activation rules
        rules.append({
            'name': 'low_intensity_activation',
            'condition': lambda m, c: (
                c.get('intensity_context') == 'low' and
                m.get('avg_path_length', 3.0) > 2.5
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
                expected_impact={'avg_path_length': -0.15, 'density': 0.10},
                context_specificity=0.85
            )
        })
        
        # Connectivity crisis rules
        rules.append({
            'name': 'connectivity_crisis',
            'condition': lambda m, c: (
                m.get('clustering_coefficient', 0) < self.thresholds.get('clustering_coefficient', {}).get('critical', 0.2) and
                m.get('density', 0) < 0.4
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
                expected_impact={'clustering_coefficient': 0.20, 'density': 0.12},
                context_specificity=0.8
            )
        })
        
        # Temporal consistency rules
        rules.append({
            'name': 'maintain_successful_pattern',
            'condition': lambda m, c: (
                self.temporal_tracker.should_maintain_consistency(c) and
                0.4 <= m.get('density', 0) <= 0.7 and
                m.get('clustering_coefficient', 0) > 0.3
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
                expected_impact={'density': 0.02, 'clustering_coefficient': 0.01},
                context_specificity=0.75
            )
        })
        
        return rules
    
    def evaluate_rules(self, metrics: Dict, context: Dict) -> List[TacticalRecommendation]:
        """Evaluate rules and return recommendations"""
        recommendations = []
        
        # Apply rules
        for rule in self.rules:
            try:
                if rule['condition'](metrics, context):
                    rec = rule['recommendation'](metrics, context)
                    if rec:
                        recommendations.append(rec)
            except Exception:
                continue
        
        # Apply context weighting and filtering
        filtered_recommendations = self._apply_context_weighting(recommendations, context)
        
        # Calculate temporal consistency
        consistency_score = self.temporal_tracker.get_consistency_score(filtered_recommendations)
        
        # Boost confidence for temporally consistent recommendations
        for rec in filtered_recommendations:
            if consistency_score > 0.7:
                rec.confidence_score = min(0.98, rec.confidence_score + 0.1)
        
        # Update temporal tracker
        self.temporal_tracker.add_recommendations(filtered_recommendations, context)
        
        return filtered_recommendations[:3]  # Return top 3
    
    def _apply_context_weighting(self, recommendations: List[TacticalRecommendation], 
                                context: Dict) -> List[TacticalRecommendation]:
        """Apply strong context weighting"""
        weighted_recommendations = []
        
        for rec in recommendations:
            rec_type = rec.recommendation_type.value
            
            # Calculate context multiplier
            context_multiplier = 1.0
            for context_type, context_value in context.items():
                if (context_type in self.context_weights and 
                    context_value in self.context_weights[context_type] and
                    rec_type in self.context_weights[context_type][context_value]):
                    
                    weight = self.context_weights[context_type][context_value][rec_type]
                    context_multiplier *= weight
            
            # Apply strong context effects
            base_confidence = rec.confidence_score
            context_effect = (context_multiplier - 1.0) * 0.4  # Strong effect
            
            # Penalty for inappropriate context
            if context_multiplier < 0.7:
                context_effect -= 0.3
            
            final_confidence = base_confidence + context_effect
            final_confidence = max(0.1, min(0.98, final_confidence))
            
            # Only include if contextually appropriate
            if context_multiplier > 0.5 and final_confidence > 0.3:
                rec.confidence_score = final_confidence
                rec.context_specificity = min(1.0, context_multiplier)
                
                # Scale expected impact by context appropriateness
                for metric in rec.expected_impact:
                    rec.expected_impact[metric] *= min(1.5, context_multiplier)
                
                weighted_recommendations.append(rec)
        
        # Sort by confidence and priority
        weighted_recommendations.sort(key=lambda x: (x.priority, -x.confidence_score))
        
        return weighted_recommendations

class TacticalRecommender:
    """Simplified tactical recommendation system"""
    
    def __init__(self, rq1_results: Dict = None):
        self.rq1_results = rq1_results  # Store RQ1 results (though we don't use them)
        self.threshold_analyzer = ThresholdAnalyzer()
        self.rule_engine = None
        self.recommendation_history = []

    
    def initialize_system(self, network_data: pd.DataFrame):
        """Initialize the recommendation system"""
        print("Initializing Tactical Recommendation System...")
        
        # Extract thresholds
        thresholds = self.threshold_analyzer.extract_thresholds(network_data)
        
        # Initialize rule engine
        self.rule_engine = RuleEngine(thresholds)
        
        print(f"✅ System initialized with {len(thresholds)} metric thresholds")
        print(f"✅ Rule engine loaded with {len(self.rule_engine.rules)} tactical rules")
        
        return self
    
    def get_recommendations(self, network_metrics: Dict, context: Dict, 
                          window_info: Dict = None) -> Dict:
        """Get tactical recommendations"""
        if self.rule_engine is None:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        # Get recommendations
        recommendations = self.rule_engine.evaluate_rules(network_metrics, context)
        
        # Analyze situation
        situation_analysis = self._analyze_situation(network_metrics, context)
        
        # Create recommendation package
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
        
        # Store in history
        self.recommendation_history.append(recommendation_package)
        
        return recommendation_package
    
    def _analyze_situation(self, metrics: Dict, context: Dict) -> Dict:
        """Analyze current tactical situation"""
        urgency_factors = []
        
        # Check critical metrics
        if metrics.get('density', 0) < 0.3:
            urgency_factors.append('critical_density')
        if metrics.get('clustering_coefficient', 0) < 0.2:
            urgency_factors.append('critical_clustering')
        
        # Context urgency
        if context.get('score_context') == 'trailing' and context.get('phase_context') == 'late':
            urgency_factors.append('desperate_situation')
        
        # Determine urgency level
        if len(urgency_factors) >= 2:
            urgency = 'very_high'
        elif len(urgency_factors) == 1:
            urgency = 'high'
        elif context.get('phase_context') == 'late':
            urgency = 'medium'
        else:
            urgency = 'normal'
        
        return {
            'urgency_level': urgency,
            'urgency_factors': urgency_factors,
            'overall_assessment': self._assess_overall_situation(metrics, context)
        }
    
    def _assess_overall_situation(self, metrics: Dict, context: Dict) -> str:
        """Assess overall tactical situation"""
        score = 0
        
        # Metric assessment
        thresholds = self.threshold_analyzer.thresholds
        
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
        
        # Context assessment
        if context.get('score_context') == 'leading':
            score += 1
        elif context.get('score_context') == 'trailing':
            score -= 1
        
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
        """Convert recommendation to dictionary"""
        return {
            'type': rec.recommendation_type.value,
            'action': rec.action,
            'confidence': rec.confidence.value,
            'confidence_score': round(rec.confidence_score, 3),
            'priority': rec.priority,
            'reasoning': rec.reasoning,
            'implementation_time': rec.implementation_time,
            'expected_impact': {k: round(v, 3) for k, v in rec.expected_impact.items()},
            'context_specificity': round(rec.context_specificity, 3),
            'triggered_metrics': rec.triggered_metrics
        }
    
    def _create_summary(self, recommendations: List[TacticalRecommendation], 
                       situation_analysis: Dict) -> Dict:
        """Create recommendation summary"""
        if not recommendations:
            return {
                'primary_focus': 'none',
                'urgency': situation_analysis['urgency_level'],
                'confidence': 0.0,
                'implementation': 'no_action'
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
        """Analyze recommendations for an entire match"""
        match_recommendations = []
        
        for idx, row in match_data.iterrows():
            # Extract network metrics
            network_metrics = {}
            for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                          'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
                if metric in row and pd.notna(row[metric]):
                    network_metrics[metric] = row[metric]
            
            # Extract context
            context = {}
            for ctx in ['score_context', 'phase_context', 'intensity_context']:
                if ctx in row and pd.notna(row[ctx]):
                    context[ctx] = row[ctx]
            
            # Window info
            window_info = {
                'window_id': idx,
                'start_minute': row.get('start_minute'),
                'end_minute': row.get('end_minute'),
                'match_id': match_id or row.get('match_id'),
                'team': row.get('team')
            }
            
            # Get recommendations
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
                'match_analysis': self._summarize_match(match_recommendations),  # Changed key name
                'match_summary': self._summarize_match(match_recommendations)    # Keep both for compatibility
            }
    
    def _summarize_match(self, match_recommendations: List[Dict]) -> Dict:
        """Summarize match-level patterns"""
        if not match_recommendations:
            return {
                'status': 'no_data',
                'critical_periods': [],
                'total_critical_windows': 0
            }
        
        # Extract patterns
        urgency_levels = [w['summary']['urgency'] for w in match_recommendations]
        primary_focuses = [w['summary']['primary_focus'] for w in match_recommendations]
        avg_confidence = np.mean([w['summary']['confidence'] for w in match_recommendations])
        
        # Find critical periods
        critical_periods = []
        for i, window in enumerate(match_recommendations):
            if window['summary']['urgency'] in ['high', 'very_high']:
                critical_periods.append({
                    'window': i,
                    'minute': window['window_info'].get('start_minute'),
                    'urgency': window['summary']['urgency'],
                    'focus': window['summary']['primary_focus']
                })
        
        return {
            'average_confidence': round(avg_confidence, 3),
            'urgency_distribution': dict(Counter(urgency_levels)),
            'focus_distribution': dict(Counter(primary_focuses)),
            'critical_periods': critical_periods,
            'total_critical_windows': len(critical_periods),
            'most_common_recommendations': dict(Counter(primary_focuses).most_common(3)),  # Add this
            'recommendation_consistency': 0.8  # Add a default value
        }

    
    def save_recommendations(self, filepath: str = "results/tactical_recommendations.json"):
        """Save recommendations to file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {
            'metadata': {
                'system_version': 'simplified_v1.0',
                'total_recommendations': len(self.recommendation_history),
                'generation_timestamp': datetime.now().isoformat()
            },
            'recommendations': self.recommendation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Recommendations saved to {filepath}")
    
    def get_system_summary(self) -> Dict:
        """Get system summary"""
        return {
            'system_status': 'initialized' if self.rule_engine else 'not_initialized',
            'system_version': 'simplified_v1.0',
            'total_rules': len(self.rule_engine.rules) if self.rule_engine else 0,
            'recommendation_history': len(self.recommendation_history),
            'temporal_tracking': 'active' if self.rule_engine else 'inactive'
        }
