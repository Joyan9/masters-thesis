import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import json
from pathlib import Path

from .threshold_analyzer import ThresholdAnalyzer
from .rule_engine import RuleEngine, TacticalRecommendation, ConfidenceLevel

class TacticalRecommender:
    """Main tactical recommendation system combining thresholds and rules"""
    
    def __init__(self, rq1_results: Dict = None):
        self.rq1_results = rq1_results
        self.threshold_analyzer = ThresholdAnalyzer(rq1_results)
        self.rule_engine = None
        self.recommendation_history = []
        
    def initialize_system(self, network_data: pd.DataFrame, 
                         outcome_column: str = 'team_performance'):
        """Initialize the recommendation system with performance thresholds"""
        
        print("Initializing Tactical Recommendation System...")
        
        # Extract thresholds from RQ1 results
        thresholds = self.threshold_analyzer.extract_performance_thresholds(
            network_data, outcome_column
        )
        
        # Initialize rule engine with thresholds
        self.rule_engine = RuleEngine(thresholds)
        
        # Save thresholds
        self.threshold_analyzer.save_thresholds()
        
        print(f"✅ System initialized with {len(thresholds)} metric thresholds")
        print(f"✅ Rule engine loaded with {len(self.rule_engine.rules)} tactical rules")
        
        return self
    
    def get_recommendations(self, network_metrics: Dict, context: Dict, 
                          window_info: Dict = None) -> Dict:
        """
        Get tactical recommendations for current situation
        
        Args:
            network_metrics: Current network metrics
            context: Current context (score, phase, intensity)
            window_info: Additional window information (time, team, etc.)
        
        Returns:
            Dictionary with recommendations and analysis
        """
        
        if self.rule_engine is None:
            raise ValueError("System not initialized. Call initialize_system() first.")
        
        # Get recommendations from rule engine
        recommendations = self.rule_engine.evaluate_rules(network_metrics, context)
        
        # Add situational analysis
        situation_analysis = self._analyze_situation(network_metrics, context)
        
        # Create recommendation package
        recommendation_package = {
            'timestamp': datetime.now().isoformat(),
            'window_info': window_info or {},
            'current_metrics': network_metrics,
            'current_context': context,
            'situation_analysis': situation_analysis,
            'recommendations': [self._recommendation_to_dict(rec) for rec in recommendations],
            'summary': self._create_recommendation_summary(recommendations, situation_analysis)
        }
        
        # Store in history
        self.recommendation_history.append(recommendation_package)
        
        return recommendation_package
    
    def _analyze_situation(self, metrics: Dict, context: Dict) -> Dict:
        """Analyze current tactical situation"""
        
        analysis = {
            'overall_assessment': 'neutral',
            'key_strengths': [],
            'key_weaknesses': [],
            'urgency_level': 'normal',
            'context_factors': []
        }
        
        # Analyze network metrics against thresholds
        metric_assessments = {}
        
        for metric_name, value in metrics.items():
            if metric_name in self.threshold_analyzer.thresholds:
                assessment = self._assess_metric_performance(metric_name, value)
                metric_assessments[metric_name] = assessment
                
                if assessment['performance'] in ['excellent', 'good']:
                    analysis['key_strengths'].append(f"{metric_name}: {assessment['description']}")
                elif assessment['performance'] in ['poor', 'critical']:
                    analysis['key_weaknesses'].append(f"{metric_name}: {assessment['description']}")
        
        # Analyze context factors
        if context.get('phase_context') == 'late':
            analysis['context_factors'].append("Late game phase - decisions more critical")
            analysis['urgency_level'] = 'high'
        
        if context.get('score_context') == 'trailing':
            analysis['context_factors'].append("Team is trailing - need attacking solutions")
            analysis['urgency_level'] = 'very_high'
            analysis['overall_assessment'] = 'concerning'
        elif context.get('score_context') == 'leading':
            analysis['context_factors'].append("Team is leading - focus on control")
            analysis['overall_assessment'] = 'positive'
        
        if context.get('intensity_context') == 'low':
            analysis['context_factors'].append("Low intensity - may need tempo increase")
        elif context.get('intensity_context') == 'high':
            analysis['context_factors'].append("High intensity - good engagement level")
        
        # Overall assessment
        weakness_count = len(analysis['key_weaknesses'])
        strength_count = len(analysis['key_strengths'])
        
        if weakness_count > strength_count + 1:
            analysis['overall_assessment'] = 'concerning'
        elif strength_count > weakness_count + 1:
            analysis['overall_assessment'] = 'positive'
        
        return analysis
    
    def _assess_metric_performance(self, metric_name: str, value: float) -> Dict:
        """Assess performance level of a specific metric"""
        
        thresholds = self.threshold_analyzer.thresholds[metric_name]['percentiles']
        
        if value >= thresholds['excellent']:
            return {
                'performance': 'excellent',
                'percentile': 90,
                'description': f"Excellent performance (top 10%)"
            }
        elif value >= thresholds['good']:
            return {
                'performance': 'good',
                'percentile': 75,
                'description': f"Good performance (top 25%)"
            }
        elif value >= thresholds['average']:
            return {
                'performance': 'average',
                'percentile': 50,
                'description': f"Average performance (median level)"
            }
        elif value >= thresholds['poor']:
            return {
                'performance': 'poor',
                'percentile': 25,
                'description': f"Below average performance (bottom 25%)"
            }
        else:
            return {
                'performance': 'critical',
                'percentile': 10,
                'description': f"Critical performance (bottom 10%)"
            }
    
    def _recommendation_to_dict(self, rec: TacticalRecommendation) -> Dict:
        """Convert recommendation object to dictionary"""
        
        return {
            'action': rec.action,
            'type': rec.recommendation_type.value,
            'confidence': rec.confidence.value,
            'confidence_score': round(rec.confidence_score, 3),
            'context': rec.context,
            'triggered_metrics': rec.triggered_metrics,
            'reasoning': rec.reasoning,
            'priority': rec.priority,
            'implementation_time': rec.implementation_time
        }
    
    def _create_recommendation_summary(self, recommendations: List[TacticalRecommendation], 
                                     situation_analysis: Dict) -> Dict:
        """Create summary of recommendations"""
        
        if not recommendations:
            return {
                'total_recommendations': 0,
                'primary_focus': 'No specific recommendations',
                'urgency': situation_analysis['urgency_level'],
                'key_message': 'Current performance appears stable'
            }
        
        # Get primary recommendation
        primary_rec = recommendations[0]
        
        # Count by type
        type_counts = {}
        for rec in recommendations:
            rec_type = rec.recommendation_type.value
            type_counts[rec_type] = type_counts.get(rec_type, 0) + 1
        
        # Determine primary focus
        primary_focus = max(type_counts.items(), key=lambda x: x[1])[0]
        
        # Create key message
        if situation_analysis['urgency_level'] == 'very_high':
            key_message = f"URGENT: {primary_rec.action}"
        elif situation_analysis['urgency_level'] == 'high':
            key_message = f"Important: {primary_rec.action}"
        else:
            key_message = f"Suggested: {primary_rec.action}"
        
        return {
            'total_recommendations': len(recommendations),
            'primary_focus': primary_focus,
            'urgency': situation_analysis['urgency_level'],
            'key_message': key_message,
            'confidence_range': {
                'highest': max(rec.confidence_score for rec in recommendations),
                'lowest': min(rec.confidence_score for rec in recommendations)
            },
            'recommendation_types': type_counts
        }
    
    def analyze_match_recommendations(self, match_data: pd.DataFrame, 
                                    match_id: str = None) -> Dict:
        """Analyze recommendations for an entire match"""
        
        match_recommendations = []
        
        for idx, row in match_data.iterrows():
            # Extract network metrics
            network_metrics = {
                'density': row.get('density'),
                'clustering_coefficient': row.get('clustering_coefficient'),
                'avg_betweenness_centrality': row.get('avg_betweenness_centrality'),
                'avg_eigenvector_centrality': row.get('avg_eigenvector_centrality'),
                'avg_path_length': row.get('avg_path_length'),
                'centralization': row.get('centralization')
            }
            
            # Extract context
            context = {
                'score_context': row.get('score_context'),
                'phase_context': row.get('phase_context'),
                'intensity_context': row.get('intensity_context')
            }
            
            # Window info
            window_info = {
                'window_id': idx,
                'start_minute': row.get('start_minute'),
                'end_minute': row.get('end_minute'),
                'match_id': match_id or row.get('match_id'),
                'team': row.get('team')
            }
            
            # Get recommendations
            window_recommendations = self.get_recommendations(
                network_metrics, context, window_info
            )
            
            match_recommendations.append(window_recommendations)
        
        # Analyze match-level patterns
        match_analysis = self._analyze_match_patterns(match_recommendations)
        
        return {
            'match_id': match_id,
            'total_windows': len(match_recommendations),
            'window_recommendations': match_recommendations,
            'match_analysis': match_analysis
        }
    
    def _analyze_match_patterns(self, match_recommendations: List[Dict]) -> Dict:
        """Analyze patterns across a full match"""
        
        # Extract recommendation types over time
        recommendation_timeline = []
        urgency_timeline = []
        
        for window_rec in match_recommendations:
            if window_rec['recommendations']:
                primary_type = window_rec['recommendations'][0]['type']
                recommendation_timeline.append(primary_type)
            else:
                recommendation_timeline.append('none')
            
            urgency_timeline.append(window_rec['summary']['urgency'])
        
        # Find critical periods
        critical_periods = []
        for i, urgency in enumerate(urgency_timeline):
            if urgency in ['high', 'very_high']:
                critical_periods.append({
                    'window': i,
                    'minute': match_recommendations[i]['window_info'].get('start_minute'),
                    'urgency': urgency,
                    'primary_recommendation': recommendation_timeline[i]
                })
        
        # Most common recommendation types
        from collections import Counter
        rec_type_counts = Counter(recommendation_timeline)
        
        return {
            'critical_periods': critical_periods,
            'most_common_recommendations': dict(rec_type_counts.most_common(3)),
            'urgency_distribution': dict(Counter(urgency_timeline)),
            'total_critical_windows': len(critical_periods),
            'recommendation_consistency': self._calculate_consistency(recommendation_timeline)
        }
    
    def _calculate_consistency(self, timeline: List[str]) -> float:
        """Calculate consistency of recommendations over time"""
        
        if len(timeline) <= 1:
            return 1.0
        
        changes = sum(1 for i in range(1, len(timeline)) 
                     if timeline[i] != timeline[i-1])
        
        # Normalize by possible changes
        max_changes = len(timeline) - 1
        consistency = 1 - (changes / max_changes) if max_changes > 0 else 1.0
        
        return round(consistency, 3)
    
    def save_recommendations(self, filepath: str = "results/tactical_recommendations.json"):
        """Save recommendation history to file"""
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.recommendation_history, f, indent=2)
        
        print(f"Recommendations saved to {filepath}")
    
    def get_system_summary(self) -> Dict:
        """Get summary of the recommendation system"""
        
        threshold_summary = self.threshold_analyzer.get_threshold_summary()
        rule_summary = self.rule_engine.get_rule_summary() if self.rule_engine else {}
        
        return {
            'system_status': 'initialized' if self.rule_engine else 'not_initialized',
            'thresholds': threshold_summary,
            'rules': rule_summary,
            'recommendation_history': {
                'total_recommendations': len(self.recommendation_history),
                'last_recommendation': self.recommendation_history[-1]['timestamp'] if self.recommendation_history else None
            }
        }
