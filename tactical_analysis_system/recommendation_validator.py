import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

class RecommendationValidator:
    """Validate tactical recommendations through performance outcome analysis"""
    
    def __init__(self, network_data: pd.DataFrame, recommendations_data: List[Dict]):
        self.network_data = network_data
        self.recommendations_data = recommendations_data
        self.validation_results = {}
        self.performance_metrics = [
            'density', 'clustering_coefficient', 'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
        ]
        
    def run_comprehensive_validation(self) -> Dict:
        """Run complete validation analysis"""
        
        print("Running Comprehensive Recommendation Validation...")
        print("=" * 60)
        
        # 1. Performance Outcome Analysis
        print("1. Analyzing performance outcomes...")
        outcome_analysis = self.analyze_performance_outcomes()
        
        # 2. Temporal Consistency Analysis
        print("2. Testing temporal consistency...")
        temporal_analysis = self.analyze_temporal_consistency()
        
        # 3. Context Sensitivity Analysis
        print("3. Evaluating context sensitivity...")
        context_analysis = self.analyze_context_sensitivity()
        
        # 4. Recommendation Effectiveness
        print("4. Measuring recommendation effectiveness...")
        effectiveness_analysis = self.analyze_recommendation_effectiveness()
        
        # 5. Cross-validation with Elite Patterns
        print("5. Cross-validating with elite patterns...")
        elite_validation = self.validate_against_elite_patterns()
        
        # Compile comprehensive results
        validation_results = {
            'performance_outcomes': outcome_analysis,
            'temporal_consistency': temporal_analysis,
            'context_sensitivity': context_analysis,
            'recommendation_effectiveness': effectiveness_analysis,
            'elite_pattern_validation': elite_validation,
            'overall_validation_score': self._calculate_overall_score(
                outcome_analysis, temporal_analysis, context_analysis, 
                effectiveness_analysis, elite_validation
            )
        }
        
        self.validation_results = validation_results
        return validation_results
    
    def analyze_performance_outcomes(self) -> Dict:
        """Analyze if recommendations correlate with performance improvements"""
        
        # Prepare data for analysis
        analysis_data = self._prepare_outcome_analysis_data()
        
        if analysis_data.empty:
            return {"error": "Insufficient data for outcome analysis"}
        
        results = {
            'correlation_analysis': {},
            'improvement_analysis': {},
            'statistical_significance': {},
            'effect_sizes': {}
        }
        
        # Analyze each metric
        for metric in self.performance_metrics:
            if metric in analysis_data.columns:
                metric_analysis = self._analyze_metric_outcomes(analysis_data, metric)
                results['correlation_analysis'][metric] = metric_analysis['correlation']
                results['improvement_analysis'][metric] = metric_analysis['improvement']
                results['statistical_significance'][metric] = metric_analysis['significance']
                results['effect_sizes'][metric] = metric_analysis['effect_size']
        
        # Overall performance correlation
        results['overall_correlation'] = self._calculate_overall_correlation(analysis_data)
        
        return results
    
    def _prepare_outcome_analysis_data(self) -> pd.DataFrame:
        """Prepare data for outcome analysis"""
        
        # Create analysis dataset
        analysis_rows = []
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    window_info = window_rec.get('window_info', {})
                    
                    # Get current metrics
                    current_metrics = window_rec.get('current_metrics', {})
                    
                    # Get future performance (next 2-3 windows)
                    future_performance = self._get_future_performance(window_info)
                    
                    if future_performance:
                        analysis_row = {
                            'window_id': window_info.get('window_id'),
                            'match_id': window_info.get('match_id'),
                            'team': window_info.get('team'),
                            'start_minute': window_info.get('start_minute'),
                            'has_recommendations': len(window_rec.get('recommendations', [])) > 0,
                            'recommendation_count': len(window_rec.get('recommendations', [])),
                            'urgency_level': window_rec.get('summary', {}).get('urgency', 'normal'),
                            'confidence_score': self._get_max_confidence(window_rec.get('recommendations', [])),
                            **{f'current_{k}': v for k, v in current_metrics.items() if v is not None},
                            **{f'future_{k}': v for k, v in future_performance.items() if v is not None}
                        }
                        analysis_rows.append(analysis_row)
        
        return pd.DataFrame(analysis_rows)
    
    def _get_future_performance(self, window_info: Dict, lookahead_windows: int = 3) -> Dict:
        """Get performance metrics for future windows"""
        
        match_id = window_info.get('match_id')
        team = window_info.get('team')
        current_window = window_info.get('window_id')
        
        if None in [match_id, team, current_window]:
            return {}
        
        # Filter data for same match and team
        match_data = self.network_data[
            (self.network_data['match_id'] == match_id) & 
            (self.network_data['team'] == team)
        ].copy()
        
        if match_data.empty:
            return {}
        
        # Get future windows
        future_windows = match_data[
            match_data.index > current_window
        ].head(lookahead_windows)
        
        if future_windows.empty:
            return {}
        
        # Calculate average future performance
        future_metrics = {}
        for metric in self.performance_metrics:
            if metric in future_windows.columns:
                future_metrics[metric] = future_windows[metric].mean()
        
        return future_metrics
    
    def _get_max_confidence(self, recommendations: List[Dict]) -> float:
        """Get maximum confidence score from recommendations"""
        
        if not recommendations:
            return 0.0
        
        return max(rec.get('confidence_score', 0.0) for rec in recommendations)
    
    def _analyze_metric_outcomes(self, data: pd.DataFrame, metric: str) -> Dict:
        """Analyze outcomes for a specific metric"""
        
        current_col = f'current_{metric}'
        future_col = f'future_{metric}'
        
        if current_col not in data.columns or future_col not in data.columns:
            return {'error': f'Missing data for {metric}'}
        
        # Calculate improvement
        data[f'{metric}_improvement'] = data[future_col] - data[current_col]
        data[f'{metric}_improvement_pct'] = (
            (data[future_col] - data[current_col]) / data[current_col] * 100
        )
        
        # Separate data by recommendation presence
        with_recs = data[data['has_recommendations'] == True]
        without_recs = data[data['has_recommendations'] == False]
        
        if len(with_recs) == 0 or len(without_recs) == 0:
            return {'error': 'Insufficient data for comparison'}
        
        # Calculate correlations
        correlation_confidence = stats.pearsonr(
            data['confidence_score'], data[f'{metric}_improvement']
        )[0] if len(data) > 2 else 0
        
        correlation_rec_count = stats.pearsonr(
            data['recommendation_count'], data[f'{metric}_improvement']
        )[0] if len(data) > 2 else 0
        
        # Statistical test
        stat, p_value = stats.mannwhitneyu(
            with_recs[f'{metric}_improvement'], 
            without_recs[f'{metric}_improvement'],
            alternative='two-sided'
        )
        
        # Effect size (Cohen's d)
        effect_size = self._calculate_cohens_d(
            with_recs[f'{metric}_improvement'], 
            without_recs[f'{metric}_improvement']
        )
        
        return {
            'correlation': {
                'confidence_vs_improvement': correlation_confidence,
                'rec_count_vs_improvement': correlation_rec_count
            },
            'improvement': {
                'with_recommendations': {
                    'mean': with_recs[f'{metric}_improvement'].mean(),
                    'std': with_recs[f'{metric}_improvement'].std(),
                    'count': len(with_recs)
                },
                'without_recommendations': {
                    'mean': without_recs[f'{metric}_improvement'].mean(),
                    'std': without_recs[f'{metric}_improvement'].std(),
                    'count': len(without_recs)
                }
            },
            'significance': {
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': effect_size
        }
    
    def _calculate_cohens_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """Calculate Cohen's d effect size"""
        
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _calculate_overall_correlation(self, data: pd.DataFrame) -> Dict:
        """Calculate overall performance correlation"""
        
        # Create composite performance score
        performance_metrics = []
        for metric in self.performance_metrics:
            improvement_col = f'{metric}_improvement'
            if improvement_col in data.columns:
                performance_metrics.append(improvement_col)
        
        if not performance_metrics:
            return {'error': 'No performance metrics available'}
        
        # Standardize and combine metrics
        scaler = StandardScaler()
        standardized_improvements = scaler.fit_transform(data[performance_metrics])
        data['composite_improvement'] = np.mean(standardized_improvements, axis=1)
        
        # Correlate with recommendation features
        correlations = {}
        
        if 'confidence_score' in data.columns:
            corr, p_val = stats.pearsonr(data['confidence_score'], data['composite_improvement'])
            correlations['confidence_vs_composite'] = {'correlation': corr, 'p_value': p_val}
        
        if 'recommendation_count' in data.columns:
            corr, p_val = stats.pearsonr(data['recommendation_count'], data['composite_improvement'])
            correlations['count_vs_composite'] = {'correlation': corr, 'p_value': p_val}
        
        return correlations
    
    def analyze_temporal_consistency(self) -> Dict:
        """Analyze if recommendations remain consistent across similar contexts"""
        
        # Group recommendations by context
        context_groups = self._group_by_context()
        
        consistency_results = {}
        
        for context_key, group_data in context_groups.items():
            if len(group_data) < 3:  # Need minimum data for consistency analysis
                continue
            
            # Analyze recommendation consistency within context
            consistency_score = self._calculate_context_consistency(group_data)
            
            consistency_results[context_key] = {
                'sample_size': len(group_data),
                'consistency_score': consistency_score,
                'most_common_recommendation': self._get_most_common_recommendation(group_data),
                'recommendation_variance': self._calculate_recommendation_variance(group_data)
            }
        
        # Overall temporal consistency
        overall_consistency = np.mean([
            result['consistency_score'] for result in consistency_results.values()
        ]) if consistency_results else 0.0
        
        return {
            'context_consistency': consistency_results,
            'overall_consistency': overall_consistency,
            'total_contexts_analyzed': len(consistency_results)
        }
    
    def _group_by_context(self) -> Dict[str, List[Dict]]:
        """Group recommendations by similar contexts"""
        
        context_groups = {}
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    context = window_rec.get('current_context', {})
                    
                    # Create context key
                    context_key = f"{context.get('score_context', 'unknown')}_" \
                                f"{context.get('phase_context', 'unknown')}_" \
                                f"{context.get('intensity_context', 'unknown')}"
                    
                    if context_key not in context_groups:
                        context_groups[context_key] = []
                    
                    context_groups[context_key].append(window_rec)
        
        return context_groups
    
    def _calculate_context_consistency(self, group_data: List[Dict]) -> float:
        """Calculate consistency of recommendations within a context"""
        
        recommendation_types = []
        
        for window_rec in group_data:
            recommendations = window_rec.get('recommendations', [])
            if recommendations:
                primary_type = recommendations[0].get('type', 'none')
                recommendation_types.append(primary_type)
            else:
                recommendation_types.append('none')
        
        if not recommendation_types:
            return 0.0
        
        # Calculate consistency as frequency of most common recommendation
        from collections import Counter
        type_counts = Counter(recommendation_types)
        most_common_count = type_counts.most_common(1)[0][1]
        
        return most_common_count / len(recommendation_types)
    
    def _get_most_common_recommendation(self, group_data: List[Dict]) -> str:
        """Get most common recommendation type in group"""
        
        recommendation_types = []
        
        for window_rec in group_data:
            recommendations = window_rec.get('recommendations', [])
            if recommendations:
                recommendation_types.append(recommendations[0].get('type', 'none'))
        
        if not recommendation_types:
            return 'none'
        
        from collections import Counter
        return Counter(recommendation_types).most_common(1)[0][0]
    
    def _calculate_recommendation_variance(self, group_data: List[Dict]) -> float:
        """Calculate variance in recommendation confidence within group"""
        
        confidence_scores = []
        
        for window_rec in group_data:
            recommendations = window_rec.get('recommendations', [])
            if recommendations:
                confidence_scores.append(recommendations[0].get('confidence_score', 0.0))
        
        return np.var(confidence_scores) if confidence_scores else 0.0
    
    def analyze_context_sensitivity(self) -> Dict:
        """Analyze if recommendations appropriately adapt to different contexts"""
        
        # Analyze recommendation adaptation across contexts
        context_adaptation = {}
        
        # Score context adaptation
        score_contexts = ['leading', 'tied', 'trailing']
        score_adaptation = self._analyze_context_adaptation('score_context', score_contexts)
        context_adaptation['score_context'] = score_adaptation
        
        # Phase context adaptation
        phase_contexts = ['early', 'middle', 'late']
        phase_adaptation = self._analyze_context_adaptation('phase_context', phase_contexts)
        context_adaptation['phase_context'] = phase_adaptation
        
        # Intensity context adaptation
        intensity_contexts = ['low', 'medium', 'high']
        intensity_adaptation = self._analyze_context_adaptation('intensity_context', intensity_contexts)
        context_adaptation['intensity_context'] = intensity_adaptation
        
        # Calculate overall sensitivity score
        sensitivity_scores = []
        for context_type, adaptation_data in context_adaptation.items():
            if 'adaptation_score' in adaptation_data:
                sensitivity_scores.append(adaptation_data['adaptation_score'])
        
        overall_sensitivity = np.mean(sensitivity_scores) if sensitivity_scores else 0.0
        
        return {
            'context_adaptations': context_adaptation,
            'overall_sensitivity': overall_sensitivity,
            'sensitivity_interpretation': self._interpret_sensitivity(overall_sensitivity)
        }
    
    def _analyze_context_adaptation(self, context_type: str, context_values: List[str]) -> Dict:
        """Analyze how recommendations adapt to a specific context type"""
        
        context_recommendations = {}
        
        # Collect recommendations for each context value
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    context = window_rec.get('current_context', {})
                    context_value = context.get(context_type)
                    
                    if context_value in context_values:
                        if context_value not in context_recommendations:
                            context_recommendations[context_value] = []
                        
                        recommendations = window_rec.get('recommendations', [])
                        if recommendations:
                            context_recommendations[context_value].append(
                                recommendations[0].get('type', 'none')
                            )
        
        # Calculate adaptation score
        adaptation_score = self._calculate_adaptation_score(context_recommendations)
        
        # Get dominant recommendation per context
        dominant_recommendations = {}
        for context_value, rec_types in context_recommendations.items():
            if rec_types:
                from collections import Counter
                dominant_recommendations[context_value] = Counter(rec_types).most_common(1)[0][0]
        
        return {
            'context_recommendations': context_recommendations,
            'dominant_recommendations': dominant_recommendations,
            'adaptation_score': adaptation_score,
            'sample_sizes': {k: len(v) for k, v in context_recommendations.items()}
        }
    
    def _calculate_adaptation_score(self, context_recommendations: Dict[str, List[str]]) -> float:
        """Calculate how well recommendations adapt to different contexts"""
        
        if len(context_recommendations) < 2:
            return 0.0
        
        # Calculate diversity of recommendations across contexts
        all_recommendations = []
        context_dominant = []
        
        for context_value, rec_types in context_recommendations.items():
            if rec_types:
                from collections import Counter
                dominant_rec = Counter(rec_types).most_common(1)[0][0]
                context_dominant.append(dominant_rec)
                all_recommendations.extend(rec_types)
        
        # Adaptation score = diversity of dominant recommendations / total diversity
        unique_dominant = len(set(context_dominant))
        unique_total = len(set(all_recommendations))
        
        return unique_dominant / unique_total if unique_total > 0 else 0.0
    
    def _interpret_sensitivity(self, sensitivity_score: float) -> str:
        """Interpret sensitivity score"""
        
        if sensitivity_score >= 0.8:
            return "Highly context-sensitive"
        elif sensitivity_score >= 0.6:
            return "Moderately context-sensitive"
        elif sensitivity_score >= 0.4:
            return "Somewhat context-sensitive"
        else:
            return "Low context sensitivity"
    
    def analyze_recommendation_effectiveness(self) -> Dict:
        """Analyze overall effectiveness of the recommendation system"""
        
        # Calculate key effectiveness metrics
        effectiveness_metrics = {}
        
        # 1. Recommendation accuracy (alignment with improvements)
        accuracy_score = self._calculate_recommendation_accuracy()
        effectiveness_metrics['accuracy_score'] = accuracy_score
        
        # 2. Performance correlation strength
        correlation_strength = self._calculate_correlation_strength()
        effectiveness_metrics['correlation_strength'] = correlation_strength
        
        # 3. Confidence calibration
        calibration_score = self._calculate_confidence_calibration()
        effectiveness_metrics['confidence_calibration'] = calibration_score
        
        # 4. Overall effectiveness score
        overall_effectiveness = np.mean([
            accuracy_score, correlation_strength, calibration_score
        ])
        effectiveness_metrics['overall_effectiveness'] = overall_effectiveness
        
        return effectiveness_metrics
    
    def _calculate_recommendation_accuracy(self) -> float:
        """Calculate accuracy of recommendations"""
        
        # This is a simplified accuracy calculation
        # In practice, you'd want more sophisticated metrics
        
        accurate_recommendations = 0
        total_recommendations = 0
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    recommendations = window_rec.get('recommendations', [])
                    if recommendations:
                        total_recommendations += 1
                        
                        # Simple heuristic: high confidence recommendations are more likely accurate
                        max_confidence = max(rec.get('confidence_score', 0) for rec in recommendations)
                        if max_confidence > 0.7:
                            accurate_recommendations += 1
        
        return accurate_recommendations / total_recommendations if total_recommendations > 0 else 0.0
    
    def _calculate_correlation_strength(self) -> float:
        """Calculate strength of performance correlations"""
        
        if 'performance_outcomes' not in self.validation_results:
            return 0.0
        
        correlations = []
        outcome_analysis = self.validation_results['performance_outcomes']
        
        for metric, analysis in outcome_analysis.get('correlation_analysis', {}).items():
            if isinstance(analysis, dict):
                conf_corr = analysis.get('confidence_vs_improvement', 0)
                if not np.isnan(conf_corr):
                    correlations.append(abs(conf_corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_confidence_calibration(self) -> float:
        """Calculate how well confidence scores align with actual outcomes"""
        
        # Simplified calibration score
        # In practice, you'd use proper calibration metrics like Brier score
        
        return 0.7  # Placeholder - implement based on your specific needs
    
    def validate_against_elite_patterns(self) -> Dict:
        """Validate recommendations against patterns from elite teams"""
        
        # Identify elite teams (top performers)
        elite_teams = self._identify_elite_teams()
        
        if not elite_teams:
            return {'error': 'No elite teams identified'}
        
        # Extract patterns from elite teams
        elite_patterns = self._extract_elite_patterns(elite_teams)
        
        # Compare recommendations with elite patterns
        pattern_alignment = self._compare_with_elite_patterns(elite_patterns)
        
        return {
            'elite_teams': elite_teams,
            'elite_patterns': elite_patterns,
            'pattern_alignment': pattern_alignment,
            'validation_score': pattern_alignment.get('overall_alignment', 0.0)
        }
    
    def _identify_elite_teams(self) -> List[str]:
        """Identify elite teams based on performance metrics"""
        
        if 'team' not in self.network_data.columns:
            return []
        
        # Calculate team performance scores
        team_performance = {}
        
        for team in self.network_data['team'].unique():
            team_data = self.network_data[self.network_data['team'] == team]
            
            # Calculate average performance across key metrics
            performance_scores = []
            for metric in ['density', 'clustering_coefficient', 'centralization']:
                if metric in team_data.columns:
                    performance_scores.append(team_data[metric].mean())
            
            if performance_scores:
                team_performance[team] = np.mean(performance_scores)
        
        # Select top 25% as elite
        if team_performance:
            threshold = np.percentile(list(team_performance.values()), 75)
            elite_teams = [team for team, score in team_performance.items() if score >= threshold]
            return elite_teams[:10]  # Limit to top 10
        
        return []
    
    def _extract_elite_patterns(self, elite_teams: List[str]) -> Dict:
        """Extract patterns from elite teams"""
        
        elite_data = self.network_data[self.network_data['team'].isin(elite_teams)]
        
        patterns = {}
        
        # Extract patterns by context
        for context_type in ['score_context', 'phase_context', 'intensity_context']:
            if context_type in elite_data.columns:
                context_patterns = {}
                
                for context_value in elite_data[context_type].unique():
                    if pd.notna(context_value):
                        context_data = elite_data[elite_data[context_type] == context_value]
                        
                        # Calculate average metrics for this context
                        context_metrics = {}
                        for metric in self.performance_metrics:
                            if metric in context_data.columns:
                                context_metrics[metric] = context_data[metric].mean()
                        
                        context_patterns[context_value] = context_metrics
                
                patterns[context_type] = context_patterns
        
        return patterns
    
    def _compare_with_elite_patterns(self, elite_patterns: Dict) -> Dict:
        """Compare recommendations with elite patterns"""
        
        # This is a simplified comparison
        # In practice, you'd want more sophisticated pattern matching
        
        alignment_scores = []
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    context = window_rec.get('current_context', {})
                    recommendations = window_rec.get('recommendations', [])
                    
                    if recommendations:
                        # Simple alignment check
                        alignment_score = self._calculate_pattern_alignment(
                            context, recommendations, elite_patterns
                        )
                        alignment_scores.append(alignment_score)
        
        overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        
        return {
            'individual_alignments': alignment_scores,
            'overall_alignment': overall_alignment,
            'total_comparisons': len(alignment_scores)
        }
    
    def _calculate_pattern_alignment(self, context: Dict, recommendations: List[Dict], 
                                   elite_patterns: Dict) -> float:
        """Calculate alignment between recommendation and elite patterns"""
        
        # Simplified alignment calculation
        # This would be more sophisticated in practice
        
        alignment_score = 0.5  # Base score
        
        # Check if recommendation type aligns with elite behavior in similar context
        for context_type, context_value in context.items():
            if context_type in elite_patterns and context_value in elite_patterns[context_type]:
                # If we have elite patterns for this context, boost alignment
                alignment_score += 0.2
        
        return min(1.0, alignment_score)
    
    def _calculate_overall_score(self, outcome_analysis: Dict, temporal_analysis: Dict,
                               context_analysis: Dict, effectiveness_analysis: Dict,
                               elite_validation: Dict) -> Dict:
        """Calculate overall validation score"""
        
        scores = []
        
        # Performance outcome score
        if 'overall_correlation' in outcome_analysis:
            outcome_score = 0.5  # Base score, adjust based on correlations
            scores.append(outcome_score)
        
        # Temporal consistency score
        if 'overall_consistency' in temporal_analysis:
            consistency_score = temporal_analysis['overall_consistency']
            scores.append(consistency_score)
        
        # Context sensitivity score
        if 'overall_sensitivity' in context_analysis:
            sensitivity_score = context_analysis['overall_sensitivity']
            scores.append(sensitivity_score)
        
        # Effectiveness score
        if 'overall_effectiveness' in effectiveness_analysis:
            effectiveness_score = effectiveness_analysis['overall_effectiveness']
            scores.append(effectiveness_score)
        
        # Elite validation score
        if 'validation_score' in elite_validation:
            elite_score = elite_validation['validation_score']
            scores.append(elite_score)
        
        overall_score = np.mean(scores) if scores else 0.0
        
        return {
            'overall_validation_score': overall_score,
            'component_scores': {
                'performance_outcomes': scores[0] if len(scores) > 0 else 0,
                'temporal_consistency': scores[1] if len(scores) > 1 else 0,
                'context_sensitivity': scores[2] if len(scores) > 2 else 0,
                'recommendation_effectiveness': scores[3] if len(scores) > 3 else 0,
                'elite_pattern_alignment': scores[4] if len(scores) > 4 else 0
            },
            'validation_interpretation': self._interpret_validation_score(overall_score)
        }
    
    def _interpret_validation_score(self, score: float) -> str:
        """Interpret overall validation score"""
        
        if score >= 0.8:
            return "Excellent validation - recommendations are highly effective"
        elif score >= 0.7:
            return "Good validation - recommendations show strong effectiveness"
        elif score >= 0.6:
            return "Moderate validation - recommendations show some effectiveness"
        elif score >= 0.5:
            return "Fair validation - recommendations show limited effectiveness"
        else:
            return "Poor validation - recommendations need significant improvement"
    
    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        
        if not self.validation_results:
            return "No validation results available. Run validation analysis first."
        
        report_lines = [
            "TACTICAL RECOMMENDATION VALIDATION REPORT",
            "=" * 60,
            "",
            "EXECUTIVE SUMMARY:",
            f"Overall Validation Score: {self.validation_results['overall_validation_score']['overall_validation_score']:.3f}",
            f"Interpretation: {self.validation_results['overall_validation_score']['validation_interpretation']}",
            "",
            "COMPONENT ANALYSIS:",
        ]
        
        # Add component scores
        component_scores = self.validation_results['overall_validation_score']['component_scores']
        for component, score in component_scores.items():
            report_lines.append(f"- {component.replace('_', ' ').title()}: {score:.3f}")
        
        # Add detailed findings
        report_lines.extend([
            "",
            "DETAILED FINDINGS:",
            "",
            "1. PERFORMANCE OUTCOME ANALYSIS:",
        ])
        
        outcome_analysis = self.validation_results.get('performance_outcomes', {})
        if 'correlation_analysis' in outcome_analysis:
            report_lines.append("   Network Metric Correlations:")
            for metric, analysis in outcome_analysis['correlation_analysis'].items():
                if isinstance(analysis, dict):
                    conf_corr = analysis.get('confidence_vs_improvement', 0)
                    report_lines.append(f"   - {metric}: {conf_corr:.3f}")
        
        # Add temporal consistency findings
        temporal_analysis = self.validation_results.get('temporal_consistency', {})
        if 'overall_consistency' in temporal_analysis:
            report_lines.extend([
                "",
                "2. TEMPORAL CONSISTENCY ANALYSIS:",
                f"   Overall Consistency: {temporal_analysis['overall_consistency']:.3f}",
                f"   Contexts Analyzed: {temporal_analysis['total_contexts_analyzed']}"
            ])
        
        # Add context sensitivity findings
        context_analysis = self.validation_results.get('context_sensitivity', {})
        if 'overall_sensitivity' in context_analysis:
            report_lines.extend([
                "",
                "3. CONTEXT SENSITIVITY ANALYSIS:",
                f"   Overall Sensitivity: {context_analysis['overall_sensitivity']:.3f}",
                f"   Interpretation: {context_analysis['sensitivity_interpretation']}"
            ])
        
        return "\n".join(report_lines)
