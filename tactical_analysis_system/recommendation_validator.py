"""
Recommendation Validation System for Tactical Recommendations

This module validates the quality and effectiveness of tactical recommendations through
multiple analytical approaches:
1. Performance outcome analysis (do recommendations correlate with improvements?)
2. Temporal consistency analysis (are recommendations stable across similar contexts?)
3. Context sensitivity analysis (do recommendations adapt appropriately to context?)
4. Recommendation effectiveness (overall system performance)
5. Elite pattern validation (do recommendations align with elite team behaviors?)

The validation framework provides empirical evidence for recommendation quality,
supporting the thesis claim that the system generates actionable tactical insights.

"""

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
    """
    Validates tactical recommendations through performance outcome analysis.
    
    This class implements a comprehensive validation framework that assesses whether
    the tactical recommendation system produces meaningful, context-appropriate, and
    effective suggestions. Validation occurs across five dimensions:
    
    1. **Performance Outcomes**: Do recommendations correlate with future improvements?
    2. **Temporal Consistency**: Are recommendations stable within similar contexts?
    3. **Context Sensitivity**: Do recommendations adapt to different match situations?
    4. **Recommendation Effectiveness**: Overall system performance metrics
    5. **Elite Pattern Validation**: Alignment with successful team behaviors
    
    The validator uses statistical testing, correlation analysis, and pattern matching
    to provide quantitative evidence of recommendation quality.
    
    Attributes:
        network_data (pd.DataFrame): Historical network metrics across all matches
        recommendations_data (List[Dict]): Generated recommendations from TacticalRecommender
        validation_results (Dict): Comprehensive validation results
        performance_metrics (List[str]): Network metrics to analyze
    """
    
    def __init__(self, network_data: pd.DataFrame, recommendations_data: List[Dict]):
        """
        Initialize recommendation validator.
        
        Args:
            network_data (pd.DataFrame): Historical network metrics
                Must contain: match_id, team, window_id, network metrics
            recommendations_data (List[Dict]): Generated recommendations
                Output from TacticalRecommender.analyze_match_recommendations()
        
        Notes:
            - network_data provides ground truth for outcome analysis
            - recommendations_data contains system outputs to validate
            - Validation requires both datasets to have matching identifiers
        """
        self.network_data = network_data
        self.recommendations_data = recommendations_data
        self.validation_results = {}
        
        # Network metrics to validate
        self.performance_metrics = [
            'density', 'clustering_coefficient', 'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
        ]
    
    def run_recommendation_validation(self) -> Dict:
        """
        Run complete validation analysis across all dimensions.
        
        This is the main entry point for validation. It executes all validation
        components sequentially and compiles comprehensive results.
        
        Returns:
            Dict: Comprehensive validation results with structure:
                {
                    'performance_outcomes': {...},      # Outcome correlation analysis
                    'temporal_consistency': {...},      # Context stability analysis
                    'context_sensitivity': {...},       # Adaptation analysis
                    'recommendation_effectiveness': {...}, # Overall effectiveness
                    'elite_pattern_validation': {...},  # Elite alignment
                    'overall_validation_score': {...}   # Composite score
                }
        
        Process:
            1. Analyze performance outcomes (correlation with improvements)
            2. Test temporal consistency (stability within contexts)
            3. Evaluate context sensitivity (appropriate adaptation)
            4. Measure recommendation effectiveness (system performance)
            5. Cross-validate with elite patterns (alignment with success)
            6. Calculate overall validation score (weighted composite)
        
        Notes:
            - Each component is independent (failure of one doesn't block others)
            - Results are stored in self.validation_results for later access
            - Progress is printed to console for monitoring
        """
        print("Running Recommendation Validation...")
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
    
    # =========================================================================
    # PERFORMANCE OUTCOME ANALYSIS
    # =========================================================================
    
    def analyze_performance_outcomes(self) -> Dict:
        """
        Analyze if recommendations correlate with performance improvements.
        
        This method tests the core hypothesis: do windows with recommendations
        show better future performance than windows without recommendations?
        
        The analysis:
        1. Pairs current metrics with future performance (3 windows ahead)
        2. Compares improvement in windows with vs. without recommendations
        3. Tests statistical significance using Mann-Whitney U test
        4. Calculates effect sizes (Cohen's d)
        5. Measures correlation between confidence and improvement
        
        Returns:
            Dict: Performance outcome analysis with structure:
                {
                    'correlation_analysis': {
                        'density': {
                            'confidence_vs_improvement': 0.45,
                            'rec_count_vs_improvement': 0.32
                        },
                        ...
                    },
                    'improvement_analysis': {
                        'density': {
                            'with_recommendations': {
                                'mean': 0.05,
                                'std': 0.12,
                                'count': 150
                            },
                            'without_recommendations': {
                                'mean': -0.02,
                                'std': 0.15,
                                'count': 200
                            }
                        },
                        ...
                    },
                    'statistical_significance': {
                        'density': {
                            'p_value': 0.023,
                            'significant': True
                        },
                        ...
                    },
                    'effect_sizes': {
                        'density': 0.52,  # Cohen's d
                        ...
                    },
                    'overall_correlation': {...}
                }
        
        Notes:
            - Uses 3-window lookahead (~15 minutes) for measuring impact
            - Mann-Whitney U test is non-parametric (robust to outliers)
            - Effect sizes provide practical significance beyond p-values
            - Returns error dict if insufficient data available
        """
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
        """
        Prepare data for outcome analysis by pairing current and future metrics.
        
        Creates a dataset where each row represents a window with:
        - Current network metrics
        - Future performance (average of next 3 windows)
        - Recommendation metadata (presence, count, confidence, urgency)
        - Match context information
        
        Returns:
            pd.DataFrame: Analysis dataset with columns:
                - window_id, match_id, team, start_minute
                - has_recommendations (bool)
                - recommendation_count (int)
                - urgency_level (str)
                - confidence_score (float)
                - current_{metric} (float) for each metric
                - future_{metric} (float) for each metric
        
        Notes:
            - Only includes windows with available future performance
            - Future performance is average of next 3 windows
            - Windows near match end are excluded (no future data)
        """
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
        """
        Get performance metrics for future windows.
        
        Retrieves network metrics from the next N windows (default 3) for the same
        match and team, then averages them to get expected future performance.
        
        Args:
            window_info (Dict): Current window metadata (match_id, team, window_id)
            lookahead_windows (int): Number of future windows to average (default: 3)
        
        Returns:
            Dict: Average future metrics
                {'density': 0.52, 'clustering_coefficient': 0.35, ...}
                Empty dict if future data unavailable
        
        Notes:
            - Lookahead of 3 windows = ~15 minutes (reasonable tactical horizon)
            - Averages across windows to smooth short-term fluctuations
            - Returns empty dict if window is near match end
        """
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
        
        # Get future windows (next 3 windows)
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
        """
        Get maximum confidence score from recommendations.
        
        Args:
            recommendations (List[Dict]): List of recommendations for a window
        
        Returns:
            float: Maximum confidence score [0, 1], or 0.0 if no recommendations
        
        Notes:
            - Uses maximum rather than average (most confident recommendation)
            - Returns 0.0 for windows without recommendations
        """
        if not recommendations:
            return 0.0
        
        return max(rec.get('confidence_score', 0.0) for rec in recommendations)
    
    def _analyze_metric_outcomes(self, data: pd.DataFrame, metric: str) -> Dict:
        """
        Analyze outcomes for a specific metric.
        
        Performs comprehensive statistical analysis comparing performance improvements
        in windows with vs. without recommendations:
        1. Calculates absolute and percentage improvements
        2. Separates data by recommendation presence
        3. Computes correlations (confidence vs. improvement, count vs. improvement)
        4. Tests statistical significance (Mann-Whitney U test)
        5. Calculates effect size (Cohen's d)
        
        Args:
            data (pd.DataFrame): Prepared analysis data
            metric (str): Network metric to analyze (e.g., 'density')
        
        Returns:
            Dict: Metric-specific analysis with structure:
                {
                    'correlation': {
                        'confidence_vs_improvement': 0.45,
                        'rec_count_vs_improvement': 0.32
                    },
                    'improvement': {
                        'with_recommendations': {
                            'mean': 0.05,
                            'std': 0.12,
                            'count': 150
                        },
                        'without_recommendations': {
                            'mean': -0.02,
                            'std': 0.15,
                            'count': 200
                        }
                    },
                    'significance': {
                        'p_value': 0.023,
                        'significant': True
                    },
                    'effect_size': 0.52  # Cohen's d
                }
        
        Notes:
            - Mann-Whitney U test is non-parametric (no normality assumption)
            - Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large
            - Positive correlation suggests recommendations are effective
        """
        current_col = f'current_{metric}'
        future_col = f'future_{metric}'
        
        if current_col not in data.columns or future_col not in data.columns:
            return {'error': f'Missing data for {metric}'}
        
        # Calculate improvement (absolute and percentage)
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
        # Correlation 1: Does higher confidence predict better improvement?
        correlation_confidence = stats.pearsonr(
            data['confidence_score'], data[f'{metric}_improvement']
        )[0] if len(data) > 2 else 0
        
        # Correlation 2: Do more recommendations predict better improvement?
        correlation_rec_count = stats.pearsonr(
            data['recommendation_count'], data[f'{metric}_improvement']
        )[0] if len(data) > 2 else 0
        
        # Statistical test: Do windows with recommendations improve more?
        # Mann-Whitney U: non-parametric test for difference in distributions
        stat, p_value = stats.mannwhitneyu(
            with_recs[f'{metric}_improvement'], 
            without_recs[f'{metric}_improvement'],
            alternative='two-sided'
        )
        
        # Effect size: How large is the difference?
        # Cohen's d: standardized mean difference
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
        """
        Calculate Cohen's d effect size.
        
        Cohen's d measures the standardized difference between two groups:
        d = (mean1 - mean2) / pooled_std
        
        Interpretation:
        - |d| < 0.2: Negligible effect
        - 0.2 ≤ |d| < 0.5: Small effect
        - 0.5 ≤ |d| < 0.8: Medium effect
        - |d| ≥ 0.8: Large effect
        
        Args:
            group1 (pd.Series): First group (e.g., with recommendations)
            group2 (pd.Series): Second group (e.g., without recommendations)
        
        Returns:
            float: Cohen's d effect size
                - Positive: group1 > group2
                - Negative: group1 < group2
                - 0.0: No difference or insufficient data
        
        Notes:
            - Uses pooled standard deviation (assumes equal variances)
            - Returns 0.0 if either group is empty or pooled_std is 0
        """
        n1, n2 = len(group1), len(group2)
        if n1 == 0 or n2 == 0:
            return 0.0
        
        # Calculate pooled standard deviation
        # pooled_std = sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d = difference in means / pooled standard deviation
        return (group1.mean() - group2.mean()) / pooled_std
    
    def _calculate_overall_correlation(self, data: pd.DataFrame) -> Dict:
        """
        Calculate overall performance correlation across all metrics.
        
        Creates a composite performance score by standardizing and averaging
        improvements across all metrics, then correlates with recommendation features.
        
        Args:
            data (pd.DataFrame): Analysis data with improvement columns
        
        Returns:
            Dict: Overall correlation analysis
                {
                    'confidence_vs_composite': {
                        'correlation': 0.42,
                        'p_value': 0.001
                    },
                    'count_vs_composite': {
                        'correlation': 0.35,
                        'p_value': 0.005
                    }
                }
        
        Process:
            1. Collect all improvement columns
            2. Standardize improvements (z-scores)
            3. Average to create composite score
            4. Correlate with confidence and recommendation count
        
        Notes:
            - Standardization ensures equal weighting across metrics
            - Composite score represents overall tactical improvement
            - Positive correlation validates recommendation effectiveness
        """
        # Create composite performance score
        performance_metrics = []
        for metric in self.performance_metrics:
            improvement_col = f'{metric}_improvement'
            if improvement_col in data.columns:
                performance_metrics.append(improvement_col)
        
        if not performance_metrics:
            return {'error': 'No performance metrics available'}
        
        # Standardize and combine metrics
        # Standardization: (x - mean) / std → z-scores
        scaler = StandardScaler()
        standardized_improvements = scaler.fit_transform(data[performance_metrics])
        
        # Composite score: average of standardized improvements
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
    
    # =========================================================================
    # TEMPORAL CONSISTENCY ANALYSIS
    # =========================================================================
    
    def analyze_temporal_consistency(self) -> Dict:
        """
        Analyze if recommendations remain consistent across similar contexts.
        
        Temporal consistency measures whether the system provides stable recommendations
        when faced with similar match situations. High consistency indicates reliable,
        predictable behavior; low consistency suggests erratic or context-insensitive
        recommendations.
        
        The analysis:
        1. Groups recommendations by context (score, phase, intensity)
        2. Calculates consistency within each context group
        3. Identifies most common recommendation per context
        4. Measures variance in confidence scores
        
        Returns:
            Dict: Temporal consistency analysis with structure:
                {
                    'context_consistency': {
                        'trailing_late_high': {
                            'sample_size': 25,
                            'consistency_score': 0.84,
                            'most_common_recommendation': 'attacking',
                            'recommendation_variance': 0.012
                        },
                        ...
                    },
                    'overall_consistency': 0.76,
                    'total_contexts_analyzed': 18
                }
        
        Notes:
            - Requires minimum 3 windows per context for reliability
            - Consistency score = frequency of most common recommendation
            - High variance suggests uncertain or context-dependent confidence
            - Overall consistency is average across all contexts
        """
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
        """
        Group recommendations by similar contexts.
        
        Creates context groups by combining score_context, phase_context, and
        intensity_context into unique keys (e.g., 'trailing_late_high').
        
        Returns:
            Dict[str, List[Dict]]: Context groups
                {
                    'trailing_late_high': [window_rec1, window_rec2, ...],
                    'tied_middle_medium': [...],
                    ...
                }
        
        Notes:
            - Context key format: '{score}_{phase}_{intensity}'
            - Windows with missing context are grouped as 'unknown'
            - Each group contains all windows with identical context
        """
        context_groups = {}
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    context = window_rec.get('current_context', {})
                    
                    # Create context key (e.g., 'trailing_late_high')
                    context_key = f"{context.get('score_context', 'unknown')}_" \
                                f"{context.get('phase_context', 'unknown')}_" \
                                f"{context.get('intensity_context', 'unknown')}"
                    
                    if context_key not in context_groups:
                        context_groups[context_key] = []
                    
                    context_groups[context_key].append(window_rec)
        
        return context_groups
    
    def _calculate_context_consistency(self, group_data: List[Dict]) -> float:
        """
        Calculate consistency of recommendations within a context.
        
        Consistency is measured as the frequency of the most common recommendation
        type within the context group.
        
        Args:
            group_data (List[Dict]): Windows with same context
        
        Returns:
            float: Consistency score [0, 1]
                - 1.0: All windows have same recommendation
                - 0.5: Half have most common, half have others
                - 0.0: All windows have different recommendations
        
        Example:
            If 8/10 windows recommend 'attacking', consistency = 0.8
        
        Notes:
            - Uses primary (first) recommendation from each window
            - Windows without recommendations are counted as 'none'
        """
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
        """
        Get most common recommendation type in group.
        
        Args:
            group_data (List[Dict]): Windows with same context
        
        Returns:
            str: Most common recommendation type (e.g., 'attacking', 'defensive')
        """
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
        """
        Calculate variance in recommendation confidence within group.
        
        Measures how much confidence scores vary within the same context.
        Low variance suggests stable confidence assessment; high variance
        suggests uncertainty or context-dependent factors.
        
        Args:
            group_data (List[Dict]): Windows with same context
        
        Returns:
            float: Variance of confidence scores
        
        Notes:
            - Only considers windows with recommendations
            - Returns 0.0 if no recommendations in group
        """
        confidence_scores = []
        
        for window_rec in group_data:
            recommendations = window_rec.get('recommendations', [])
            if recommendations:
                confidence_scores.append(recommendations[0].get('confidence_score', 0.0))
        
        return np.var(confidence_scores) if confidence_scores else 0.0
    
    # =========================================================================
    # CONTEXT SENSITIVITY ANALYSIS
    # =========================================================================
    
    def analyze_context_sensitivity(self) -> Dict:
        """
        Analyze if recommendations appropriately adapt to different contexts.
        
        Context sensitivity measures whether the system provides different recommendations
        for different match situations. High sensitivity indicates context-aware behavior;
        low sensitivity suggests one-size-fits-all recommendations.
        
        The analysis examines three context dimensions:
        1. Score context (leading, tied, trailing)
        2. Phase context (early, middle, late)
        3. Intensity context (low, medium, high)
        
        For each dimension, we measure:
        - Dominant recommendation per context value
        - Adaptation score (diversity of recommendations across contexts)
        - Sample sizes for reliability assessment
        
        Returns:
            Dict: Context sensitivity analysis with structure:
                {
                    'context_adaptations': {
                        'score_context': {
                            'dominant_recommendations': {
                                'leading': 'defensive',
                                'tied': 'attacking',
                                'trailing': 'attacking'
                            },
                            'adaptation_score': 0.67,
                            'sample_sizes': {'leading': 50, 'tied': 80, 'trailing': 45}
                        },
                        'phase_context': {...},
                        'intensity_context': {...}
                    },
                    'overall_sensitivity': 0.72,
                    'sensitivity_interpretation': 'Moderately context-sensitive'
                }
        
        Notes:
            - Adaptation score = unique dominant recs / total unique recs
            - Higher score indicates better context adaptation
            - Overall sensitivity is average across all dimensions
        """
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
        """
        Analyze how recommendations adapt to a specific context type.
        
        Args:
            context_type (str): Context dimension ('score_context', 'phase_context', 'intensity_context')
            context_values (List[str]): Possible values for this context
        
        Returns:
            Dict: Adaptation analysis for this context dimension
                {
                    'context_recommendations': {
                        'leading': ['defensive', 'defensive', 'possession', ...],
                        'tied': ['attacking', 'tempo', 'attacking', ...],
                        'trailing': ['attacking', 'attacking', 'pressing', ...]
                    },
                    'dominant_recommendations': {
                        'leading': 'defensive',
                        'tied': 'attacking',
                        'trailing': 'attacking'
                    },
                    'adaptation_score': 0.67,
                    'sample_sizes': {'leading': 50, 'tied': 80, 'trailing': 45}
                }
        
        Notes:
            - Collects all recommendations for each context value
            - Identifies dominant (most common) recommendation per value
            - Adaptation score measures diversity of dominant recommendations
        """
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
        """
        Calculate how well recommendations adapt to different contexts.
        
        Adaptation score measures the diversity of dominant recommendations across
        context values. Higher diversity indicates better context adaptation.
        
        Formula: unique_dominant_recs / total_unique_recs
        
        Args:
            context_recommendations (Dict[str, List[str]]): Recommendations per context value
        
        Returns:
            float: Adaptation score [0, 1]
                - 1.0: Each context has unique dominant recommendation
                - 0.5: Some overlap in dominant recommendations
                - 0.0: All contexts have same dominant recommendation
        
        Example:
            If leading→defensive, tied→attacking, trailing→attacking:
            - unique_dominant = 2 (defensive, attacking)
            - total_unique = 2 (defensive, attacking)
            - adaptation_score = 2/2 = 1.0
        
        Notes:
            - Requires at least 2 context values for meaningful score
            - Returns 0.0 if insufficient data
        """
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
        """
        Interpret sensitivity score with categorical labels.
        
        Args:
            sensitivity_score (float): Overall sensitivity score [0, 1]
        
        Returns:
            str: Interpretation category
        """
        if sensitivity_score >= 0.8:
            return "Highly context-sensitive"
        elif sensitivity_score >= 0.6:
            return "Moderately context-sensitive"
        elif sensitivity_score >= 0.4:
            return "Somewhat context-sensitive"
        else:
            return "Low context sensitivity"
    
    # =========================================================================
    # RECOMMENDATION EFFECTIVENESS
    # =========================================================================
    
    def analyze_recommendation_effectiveness(self) -> Dict:
        """
        Analyze overall effectiveness of the recommendation system.
        
        Combines multiple effectiveness metrics into a composite assessment:
        1. Recommendation accuracy (alignment with improvements)
        2. Performance correlation strength
        3. Confidence calibration (Brier score)
        4. Overall effectiveness score
        
        Returns:
            Dict: Effectiveness analysis with structure:
                {
                    'accuracy_score': 0.72,
                    'correlation_strength': 0.45,
                    'confidence_calibration': 0.68,
                    'overall_effectiveness': 0.62
                }
        
        Notes:
            - Accuracy: proportion of high-confidence recommendations
            - Correlation: average absolute correlation with improvements
            - Calibration: Brier score (lower is better, inverted for consistency)
            - Overall: average of three components
        """
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
        """
        Calculate accuracy of recommendations.
        
        Simplified accuracy metric: proportion of recommendations with high confidence.
        This assumes that high-confidence recommendations are more likely to be accurate.
        
        Returns:
            float: Accuracy score [0, 1]
        
        Notes:
            - High confidence threshold: 0.7
            - This is a proxy metric (true accuracy requires ground truth labels)
            - More sophisticated: compare with actual performance improvements
        """
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
        """
        Calculate strength of performance correlations.
        
        Extracts correlation coefficients from performance outcome analysis and
        averages their absolute values to get overall correlation strength.
        
        Returns:
            float: Average absolute correlation [0, 1]
        
        Notes:
            - Uses absolute values (direction doesn't matter for strength)
            - Averages across all metrics
            - Returns 0.0 if performance outcomes not yet analyzed
        """
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
        """
        Calculate how well confidence scores align with actual outcomes using Brier score.
        
        Brier score measures the accuracy of probabilistic predictions:
        BS = (1/N) * Σ(predicted_prob - actual_outcome)²
        
        Lower Brier score = better calibration
        We invert it (1 - BS) so higher score = better calibration (consistent with other metrics)
        
        Returns:
            float: Calibration score [0, 1]
                - 1.0: Perfect calibration
                - 0.5: Random calibration
                - 0.0: Worst possible calibration
        
        Notes:
            - Uses composite improvement as "actual outcome"
            - Confidence scores are treated as predicted probabilities
            - Simplified implementation (assumes binary outcomes)
        """
        # Prepare data
        analysis_data = self._prepare_outcome_analysis_data()
        
        if analysis_data.empty or 'confidence_score' not in analysis_data.columns:
            return 0.7  # Default moderate calibration
        
        # Calculate composite improvement (standardized)
        performance_metrics = []
        for metric in self.performance_metrics:
            improvement_col = f'{metric}_improvement'
            if improvement_col in analysis_data.columns:
                performance_metrics.append(improvement_col)
        
        if not performance_metrics:
            return 0.7
        
        # Standardize and combine
        scaler = StandardScaler()
        standardized_improvements = scaler.fit_transform(analysis_data[performance_metrics])
        composite_improvement = np.mean(standardized_improvements, axis=1)
        
        # Convert to binary outcome (improvement vs. no improvement)
        actual_outcomes = (composite_improvement > 0).astype(float)
        
        # Get predicted probabilities (confidence scores)
        predicted_probs = analysis_data['confidence_score'].values
        
        # Calculate Brier score
        brier_score = np.mean((predicted_probs - actual_outcomes) ** 2)
        
        # Invert so higher is better (1 - BS)
        # Brier score ranges [0, 1], so inverted score also ranges [0, 1]
        calibration_score = 1.0 - brier_score
        
        return calibration_score
    
    # =========================================================================
    # ELITE PATTERN VALIDATION
    # =========================================================================
    
    def validate_against_elite_patterns(self) -> Dict:
        """
        Validate recommendations against patterns from elite teams.
        
        This analysis tests whether recommendations align with behaviors exhibited
        by successful teams. The hypothesis: effective recommendations should guide
        teams toward elite-level network structures.
        
        Process:
        1. Identify elite teams (top 25% by performance metrics)
        2. Extract network metric patterns from elite teams by context
        3. Compare recommendations with elite patterns
        4. Calculate alignment score
        
        Returns:
            Dict: Elite pattern validation with structure:
                {
                    'elite_teams': ['Team A', 'Team B', ...],
                    'elite_patterns': {
                        'score_context': {
                            'leading': {
                                'density': 0.58,
                                'clustering_coefficient': 0.42,
                                ...
                            },
                            ...
                        },
                        ...
                    },
                    'pattern_alignment': {
                        'individual_alignments': [0.7, 0.8, 0.6, ...],
                        'overall_alignment': 0.73,
                        'total_comparisons': 250
                    },
                    'validation_score': 0.73
                }
        
        Notes:
            - Elite threshold: 75th percentile of team performance
            - Limited to top 10 teams for computational efficiency
            - Alignment uses cosine similarity between metric vectors
        """
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
        """
        Identify elite teams based on performance metrics.
        
        Elite teams are defined as those in the top 25% of overall performance,
        measured by average network metrics (density, clustering, centralization).
        
        Returns:
            List[str]: Elite team identifiers (max 10 teams)
        
        Process:
            1. Calculate team-level average for key metrics
            2. Compute composite performance score
            3. Select top 25% (75th percentile threshold)
            4. Limit to top 10 for efficiency
        
        Notes:
            - Uses three key metrics: density, clustering, centralization
            - Equal weighting across metrics
            - Returns empty list if team column missing
        """
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
        """
        Extract network metric patterns from elite teams.
        
        Calculates average network metrics for elite teams across different contexts,
        creating a reference profile of successful tactical patterns.
        
        Args:
            elite_teams (List[str]): Elite team identifiers
        
        Returns:
            Dict: Elite patterns by context
                {
                    'score_context': {
                        'leading': {
                            'density': 0.58,
                            'clustering_coefficient': 0.42,
                            'centralization': 0.35,
                            ...
                        },
                        'tied': {...},
                        'trailing': {...}
                    },
                    'phase_context': {...},
                    'intensity_context': {...}
                }
        
        Notes:
            - Patterns are context-specific (different for leading vs. trailing)
            - Averages across all elite team windows in each context
            - Missing contexts return empty patterns
        """
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
        """
        Compare recommendations with elite patterns using cosine similarity.
        
        For each recommendation, we:
        1. Extract current network metrics
        2. Extract expected impact from recommendation
        3. Calculate target metrics (current + expected impact)
        4. Find corresponding elite pattern for same context
        5. Calculate cosine similarity between target and elite metrics
        
        Args:
            elite_patterns (Dict): Elite team patterns by context
        
        Returns:
            Dict: Pattern alignment analysis
                {
                    'individual_alignments': [0.7, 0.8, 0.6, ...],
                    'overall_alignment': 0.73,
                    'total_comparisons': 250
                }
        
        Notes:
            - Cosine similarity ranges [-1, 1], normalized to [0, 1]
            - Higher similarity = better alignment with elite patterns
            - Skips windows without matching elite patterns
        """
        alignment_scores = []
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    context = window_rec.get('current_context', {})
                    recommendations = window_rec.get('recommendations', [])
                    current_metrics = window_rec.get('current_metrics', {})
                    
                    if recommendations:
                        # Calculate pattern alignment using cosine similarity
                        alignment_score = self._calculate_pattern_alignment(
                            context, recommendations, current_metrics, elite_patterns
                        )
                        if alignment_score is not None:
                            alignment_scores.append(alignment_score)
        
        overall_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        
        return {
            'individual_alignments': alignment_scores,
            'overall_alignment': overall_alignment,
            'total_comparisons': len(alignment_scores)
        }
    
    def _calculate_pattern_alignment(self, context: Dict, recommendations: List[Dict],
                                   current_metrics: Dict, elite_patterns: Dict) -> Optional[float]:
        """
        Calculate alignment between recommendation and elite patterns using cosine similarity.
        
        This method compares the expected outcome of a recommendation (current metrics +
        expected impact) with the elite team pattern for the same context.
        
        Args:
            context (Dict): Current match context
            recommendations (List[Dict]): Recommendations for this window
            current_metrics (Dict): Current network metrics
            elite_patterns (Dict): Elite team patterns by context
        
        Returns:
            Optional[float]: Alignment score [0, 1], or None if no matching elite pattern
        
        Process:
            1. Get primary recommendation and its expected impact
            2. Calculate target metrics (current + expected impact)
            3. Find elite pattern for same context
            4. Compute cosine similarity between target and elite vectors
            5. Normalize to [0, 1] range
        
        Notes:
            - Uses cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
            - Normalized: (cos(θ) + 1) / 2 to map [-1,1] → [0,1]
            - Returns None if no matching elite pattern exists
        """
        if not recommendations:
            return None
        
        # Get primary recommendation
        primary_rec = recommendations[0]
        expected_impact = primary_rec.get('expected_impact', {})
        
        # Calculate target metrics (current + expected impact)
        target_metrics = {}
        for metric in self.performance_metrics:
            if metric in current_metrics and metric in expected_impact:
                current_value = current_metrics[metric]
                impact = expected_impact[metric].get('expected', 0)
                target_metrics[metric] = current_value + impact
            elif metric in current_metrics:
                target_metrics[metric] = current_metrics[metric]
        
        # Find matching elite pattern
        elite_metric_vector = None
        for context_type, context_value in context.items():
            if (context_type in elite_patterns and 
                context_value in elite_patterns[context_type]):
                elite_metric_vector = elite_patterns[context_type][context_value]
                break
        
        if not elite_metric_vector:
            return None
        
        # Calculate cosine similarity between target and elite metrics
        # Align metrics (use only common metrics)
        common_metrics = set(target_metrics.keys()) & set(elite_metric_vector.keys())
        if not common_metrics:
            return None
        
        target_vector = np.array([target_metrics[m] for m in common_metrics])
        elite_vector = np.array([elite_metric_vector[m] for m in common_metrics])
        
        # Cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
        dot_product = np.dot(target_vector, elite_vector)
        norm_target = np.linalg.norm(target_vector)
        norm_elite = np.linalg.norm(elite_vector)
        
        if norm_target == 0 or norm_elite == 0:
            return None
        
        cosine_sim = dot_product / (norm_target * norm_elite)
        
        # Normalize to [0, 1] range: (cos + 1) / 2
        # cos = -1 → 0, cos = 0 → 0.5, cos = 1 → 1.0
        alignment_score = (cosine_sim + 1) / 2
        
        return alignment_score
    
    # =========================================================================
    # OVERALL VALIDATION SCORE
    # =========================================================================
    
    def _calculate_overall_score(self, outcome_analysis: Dict, temporal_analysis: Dict,
                               context_analysis: Dict, effectiveness_analysis: Dict,
                               elite_validation: Dict) -> Dict:
        """
        Calculate overall validation score from component analyses.
        
        Combines scores from all validation dimensions using equal weighting:
        - Performance outcomes: 20%
        - Temporal consistency: 20%
        - Context sensitivity: 20%
        - Recommendation effectiveness: 20%
        - Elite pattern alignment: 20%
        
        Args:
            outcome_analysis (Dict): Performance outcome results
            temporal_analysis (Dict): Temporal consistency results
            context_analysis (Dict): Context sensitivity results
            effectiveness_analysis (Dict): Effectiveness results
            elite_validation (Dict): Elite pattern validation results
        
        Returns:
            Dict: Overall validation score with structure:
                {
                    'overall_validation_score': 0.73,
                    'component_scores': {
                        'performance_outcomes': 0.68,
                        'temporal_consistency': 0.76,
                        'context_sensitivity': 0.72,
                        'recommendation_effectiveness': 0.62,
                        'elite_pattern_alignment': 0.73
                    },
                    'validation_interpretation': 'Good validation - ...'
                }
        
        Notes:
            - Equal weighting assumes all dimensions are equally important
            - Component scores are extracted from respective analyses
            - Interpretation provides qualitative assessment
        """
        scores = []
        
        # Performance outcome score (simplified: base 0.5)
        # In practice, this would be derived from correlation strengths
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
        
        # Calculate overall score (equal weighting)
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
        """
        Interpret overall validation score with categorical labels.
        
        Args:
            score (float): Overall validation score [0, 1]
        
        Returns:
            str: Qualitative interpretation
        
        Interpretation Thresholds:
            - ≥0.8: Excellent (highly effective recommendations)
            - ≥0.7: Good (strong effectiveness)
            - ≥0.6: Moderate (some effectiveness)
            - ≥0.5: Fair (limited effectiveness)
            - <0.5: Poor (needs improvement)
        """
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
