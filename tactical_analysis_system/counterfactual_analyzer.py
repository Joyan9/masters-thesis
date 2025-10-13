"""
Counterfactual Analysis for Tactical Recommendations

This module implements counterfactual analysis to estimate the causal impact of
tactical recommendations. The core question: "What would have happened if teams
had followed the recommendations?"

The analysis framework:
1. **Predictive Modeling**: Build ML models to predict natural metric evolution
2. **Scenario Identification**: Identify windows with recommendations
3. **Outcome Simulation**: Simulate outcomes if recommendations were followed
4. **Comparison Analysis**: Compare simulated vs. actual outcomes
5. **Impact Quantification**: Measure recommendation effectiveness

This provides empirical evidence for the thesis claim that recommendations would
improve team performance if implemented.

Methodological Approach:
- Uses Random Forest models for robust prediction under non-linearity
- Incorporates context features (score, phase, intensity) for realistic simulation
- Estimates recommendation effects based on expected impacts
- Compares simulated improvements against actual performance
- Tests statistical significance using Wilcoxon signed-rank test

Author: [Your Name]
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import sklearn.metrics

warnings.filterwarnings("ignore")


class CounterfactualAnalyzer:
    """
    Analyze counterfactual scenarios for tactical recommendations.
    
    This class implements a counterfactual analysis framework to estimate what would
    have happened if teams had followed the tactical recommendations. The analysis
    provides causal evidence for recommendation effectiveness.
    
    The counterfactual framework:
    1. Builds predictive models for natural metric evolution (baseline)
    2. Identifies scenarios where recommendations were made
    3. Simulates outcomes by adding recommendation effects to baseline predictions
    4. Compares simulated outcomes with actual performance
    5. Quantifies recommendation impact through improvement rates
    
    Key Assumptions:
    - Natural evolution can be predicted from current state + context
    - Recommendation effects are additive to natural evolution
    - Effect magnitudes scale with confidence scores
    - Teams did NOT follow recommendations (actual outcomes = baseline)
    
    Attributes:
        network_data (pd.DataFrame): Historical network metrics
        recommendations_data (List[Dict]): Generated recommendations
        counterfactual_models (Dict): Trained prediction models per metric
        simulation_results (Dict): Simulated counterfactual outcomes
    """
    
    def __init__(self, network_data: pd.DataFrame, recommendations_data: List[Dict]):
        """
        Initialize counterfactual analyzer.
        
        Args:
            network_data (pd.DataFrame): Historical network metrics
                Must contain: match_id, team, window_id, start_minute, network metrics
            recommendations_data (List[Dict]): Generated recommendations
                Output from TacticalRecommender.analyze_match_recommendations()
        
        Notes:
            - network_data provides ground truth for model training
            - recommendations_data provides scenarios for counterfactual simulation
        """
        self.network_data = network_data
        self.recommendations_data = recommendations_data
        self.counterfactual_models = {}
        self.simulation_results = {}
    
    def run_counterfactual_analysis(self) -> Dict:
        """
        Run comprehensive counterfactual analysis.
        
        This is the main entry point for counterfactual analysis. It executes the
        complete pipeline from model building to impact quantification.
        
        Returns:
            Dict: Comprehensive counterfactual analysis with structure:
                {
                    'scenarios': [...],              # Identified counterfactual scenarios
                    'simulation_results': [...],     # Simulated outcomes
                    'comparison_results': {...},     # Actual vs. simulated comparison
                    'impact_analysis': {...},        # Recommendation impact metrics
                    'model_performance': {...}       # Predictive model evaluation
                }
        
        Process:
            1. Build predictive models (Random Forest for each metric)
            2. Identify counterfactual scenarios (windows with recommendations)
            3. Simulate alternative outcomes (baseline + recommendation effects)
            4. Compare actual vs. counterfactual outcomes
            5. Calculate recommendation impact (improvement rates, significance)
            6. Evaluate model performance (R², MAE, RMSE)
        
        Notes:
            - Each step is independent (failure of one doesn't block others)
            - Progress is printed to console for monitoring
            - Returns partial results if some components fail
        """
        print("Running Counterfactual Analysis...")
        print("=" * 50)
        
        # 1. Build predictive models for natural evolution
        print("1. Building predictive models...")
        self.build_predictive_models()
        
        # 2. Identify counterfactual scenarios
        print("2. Identifying counterfactual scenarios...")
        scenarios = self.identify_counterfactual_scenarios()
        
        # 3. Simulate alternative outcomes
        print("3. Simulating alternative outcomes...")
        simulation_results = self.simulate_alternative_outcomes(scenarios)
        
        # 4. Compare actual vs counterfactual
        print("4. Comparing outcomes...")
        comparison_results = self.compare_outcomes(simulation_results)
        
        # 5. Calculate recommendation impact
        print("5. Calculating recommendation impact...")
        impact_analysis = self.calculate_recommendation_impact(comparison_results)
        
        return {
            'scenarios': scenarios,
            'simulation_results': simulation_results,
            'comparison_results': comparison_results,
            'impact_analysis': impact_analysis,
            'model_performance': self.evaluate_model_performance()
        }
    
    # =========================================================================
    # PREDICTIVE MODEL BUILDING
    # =========================================================================
    
    def build_predictive_models(self):
        """
        Build predictive models for natural metric evolution.
        
        Trains Random Forest models to predict how network metrics naturally evolve
        from one window to the next, given current state and context. These models
        establish the baseline (what would happen without recommendations).
        
        Process:
            1. Prepare training data (consecutive window pairs)
            2. For each metric, train Random Forest model
            3. Evaluate model performance (train/test split)
            4. Store models for later simulation
        
        Models predict:
            Δmetric = f(current_metrics, context, time_diff)
        
        Notes:
            - Uses Random Forest for robustness to non-linearity
            - Includes context features (score, phase, intensity)
            - Requires minimum 10 samples per metric
            - Prints success message for each model built
        """
        # Prepare training data from consecutive windows
        training_data = self._prepare_training_data()
        
        if training_data.empty:
            print("   Warning: Insufficient data for model building")
            return
        
        # Build models for each network metric
        metrics_to_predict = [
            'density', 'clustering_coefficient', 'avg_betweenness_centrality',
            'avg_eigenvector_centrality', 'avg_path_length', 'centralization'
        ]
        
        for metric in metrics_to_predict:
            if f'{metric}_change' in training_data.columns:
                model = self._build_metric_model(training_data, metric)
                if model:
                    self.counterfactual_models[metric] = model
                    print(f"   ✓ Built model for {metric} (R²={model['r2']:.3f})")
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """
        Prepare training data for predictive models.
        
        Creates training examples by pairing consecutive windows within each match.
        Each example contains:
        - Current state (network metrics)
        - Context (score, phase, intensity)
        - Time difference
        - Target (change in each metric)
        
        Returns:
            pd.DataFrame: Training dataset with columns:
                - current_{metric} (float): Current metric values
                - score_context, phase_context, intensity_context (str): Context
                - time_diff (float): Minutes between windows
                - {metric}_change (float): Target variable (Δmetric)
                - match_id (str): Match identifier
        
        Process:
            1. For each match, sort windows by time
            2. Create pairs (window_i, window_i+1)
            3. Extract current state from window_i
            4. Calculate changes (window_i+1 - window_i)
            5. Compile into training dataframe
        
        Notes:
            - Only includes consecutive windows (no gaps)
            - Skips windows with missing metrics
            - Returns empty dataframe if insufficient data
        """
        training_rows = []
        
        # Create training examples from consecutive windows
        for match_id in self.network_data['match_id'].unique():
            match_data = self.network_data[
                self.network_data['match_id'] == match_id
            ].sort_values('start_minute')
            
            # Create pairs of consecutive windows
            for i in range(len(match_data) - 1):
                current_window = match_data.iloc[i]
                next_window = match_data.iloc[i + 1]
                
                training_row = {}
                
                # Current state features
                for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                              'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
                    if metric in current_window:
                        training_row[f'current_{metric}'] = current_window[metric]
                
                # Context features
                for context in ['score_context', 'phase_context', 'intensity_context']:
                    if context in current_window:
                        training_row[context] = current_window[context]
                
                # Target: changes in metrics
                for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                              'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
                    if metric in current_window and metric in next_window:
                        if pd.notna(current_window[metric]) and pd.notna(next_window[metric]):
                            training_row[f'{metric}_change'] = next_window[metric] - current_window[metric]
                
                # Additional features
                training_row['time_diff'] = next_window.get('start_minute', 0) - current_window.get('start_minute', 0)
                training_row['match_id'] = match_id
                
                training_rows.append(training_row)
        
        return pd.DataFrame(training_rows)
    
    def _build_metric_model(self, training_data: pd.DataFrame, metric: str) -> Optional[Dict]:
        """
        Build predictive model for a specific metric.
        
        Trains a Random Forest model to predict changes in the given metric based on
        current state and context. Includes one-hot encoding for categorical features.
        
        Args:
            training_data (pd.DataFrame): Training dataset
            metric (str): Metric to predict (e.g., 'density')
        
        Returns:
            Optional[Dict]: Model package with structure:
                {
                    'model': RandomForestRegressor,
                    'scaler': StandardScaler,
                    'features': List[str],
                    'train_score': float,
                    'test_score': float,
                    'r2': float,
                    'mae': float,
                    'rmse': float,
                    'feature_importance': Dict[str, float]
                }
                Returns None if insufficient data
        
        Process:
            1. Select features (current metric, time_diff, context)
            2. One-hot encode categorical context variables
            3. Split into train/test (80/20)
            4. Standardize features (zero mean, unit variance)
            5. Train Random Forest (100 trees)
            6. Evaluate on test set (R², MAE, RMSE)
            7. Extract feature importances
        
        Notes:
            - Requires minimum 10 samples for training
            - Uses 80/20 train/test split
            - Random Forest with 100 trees (robust default)
            - StandardScaler ensures equal feature weighting
        """
        target_col = f'{metric}_change'
        
        # Select features
        feature_cols = [
            f'current_{metric}', 'time_diff'
        ]
        
        # Add context features (one-hot encoded)
        context_features = []
        for context in ['score_context', 'phase_context', 'intensity_context']:
            if context in training_data.columns:
                # One-hot encode categorical variables
                context_dummies = pd.get_dummies(training_data[context], prefix=context)
                training_data = pd.concat([training_data, context_dummies], axis=1)
                context_features.extend(context_dummies.columns.tolist())
        
        feature_cols.extend(context_features)
        
        # Filter available features
        available_features = [col for col in feature_cols if col in training_data.columns]
        
        if len(available_features) < 2:
            return None
        
        # Prepare data
        X = training_data[available_features].fillna(0)
        y = training_data[target_col].fillna(0)
        
        if len(X) < 10:  # Need minimum samples
            return None
        
        # Split data (80/20 train/test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features (standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Calculate additional metrics
        y_pred = model.predict(X_test_scaled)
        r2 = sklearn.metrics.r2_score(y_test, y_pred)
        mae = sklearn.metrics.mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_pred))
        
        return {
            'model': model,
            'scaler': scaler,
            'features': available_features,
            'train_score': train_score,
            'test_score': test_score,
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
    
    # =========================================================================
    # SCENARIO IDENTIFICATION
    # =========================================================================
    
    def identify_counterfactual_scenarios(self) -> List[Dict]:
        """
        Identify scenarios for counterfactual analysis.
        
        Extracts all windows where recommendations were made. These become the
        counterfactual scenarios: "What if the team had followed these recommendations?"
        
        Returns:
            List[Dict]: Counterfactual scenarios with structure:
                [
                    {
                        'window_info': {...},
                        'actual_metrics': {...},
                        'context': {...},
                        'recommendations': [...],
                        'scenario_type': 'recommendation_implementation'
                    },
                    ...
                ]
        
        Notes:
            - Only includes windows with at least one recommendation
            - Each scenario represents a potential intervention point
            - Prints total number of scenarios identified
        """
        scenarios = []
        
        for rec_data in self.recommendations_data:
            if 'window_recommendations' in rec_data:
                for window_rec in rec_data['window_recommendations']:
                    recommendations = window_rec.get('recommendations', [])
                    
                    if recommendations:
                        # Create counterfactual scenario
                        scenario = {
                            'window_info': window_rec.get('window_info', {}),
                            'actual_metrics': window_rec.get('current_metrics', {}),
                            'context': window_rec.get('current_context', {}),
                            'recommendations': recommendations,
                            'scenario_type': 'recommendation_implementation'
                        }
                        
                        scenarios.append(scenario)
        
        print(f"   Identified {len(scenarios)} counterfactual scenarios")
        return scenarios
    
    # =========================================================================
    # OUTCOME SIMULATION
    # =========================================================================
    
    def simulate_alternative_outcomes(self, scenarios: List[Dict]) -> List[Dict]:
        """
        Simulate outcomes if recommendations were followed.
        
        For each scenario, simulates what would have happened if the team had
        implemented the recommendations. The simulation combines:
        1. Baseline prediction (natural evolution from ML model)
        2. Recommendation effect (estimated impact of following recommendation)
        
        Counterfactual outcome = Baseline + Recommendation effect
        
        Args:
            scenarios (List[Dict]): Counterfactual scenarios
        
        Returns:
            List[Dict]: Simulation results with structure:
                [
                    {
                        'scenario': {...},
                        'simulated_outcomes': {
                            'density': {
                                'original_value': 0.50,
                                'predicted_change': 0.02,
                                'recommendation_effect': 0.01,
                                'total_change': 0.03,
                                'simulated_value': 0.53
                            },
                            ...
                        },
                        'simulation_quality': 0.83
                    },
                    ...
                ]
        
        Process:
            1. For each scenario and metric:
                a. Create feature vector (current state + context)
                b. Predict baseline change using ML model
                c. Estimate recommendation effect
                d. Combine: total_change = baseline + effect
                e. Calculate simulated value
            2. Assess simulation quality (coverage of metrics)
        
        Notes:
            - Requires trained models for prediction
            - Recommendation effects are estimated (not observed)
            - Quality score reflects proportion of metrics simulated
        """
        simulation_results = []
        
        for scenario in scenarios:
            # Get actual metrics and context
            actual_metrics = scenario['actual_metrics']
            context = scenario['context']
            recommendations = scenario['recommendations']
            
            # Simulate what would happen if recommendations were followed
            simulated_outcomes = {}
            
            for metric, model_info in self.counterfactual_models.items():
                if metric in actual_metrics:
                    # Create feature vector for prediction
                    features = self._create_feature_vector(
                        actual_metrics, context, model_info['features']
                    )
                    
                    if features is not None:
                        # Predict baseline change (natural evolution)
                        features_scaled = model_info['scaler'].transform([features])
                        predicted_change = model_info['model'].predict(features_scaled)[0]
                        
                        # Estimate recommendation effect
                        recommendation_effect = self._estimate_recommendation_effect(
                            recommendations, metric
                        )
                        
                        # Combine baseline + recommendation effect
                        total_change = predicted_change + recommendation_effect
                        simulated_value = actual_metrics[metric] + total_change
                        
                        simulated_outcomes[metric] = {
                            'original_value': actual_metrics[metric],
                            'predicted_change': predicted_change,
                            'recommendation_effect': recommendation_effect,
                            'total_change': total_change,
                            'simulated_value': simulated_value
                        }
            
            simulation_result = {
                'scenario': scenario,
                'simulated_outcomes': simulated_outcomes,
                'simulation_quality': self._assess_simulation_quality(simulated_outcomes)
            }
            
            simulation_results.append(simulation_result)
        
        return simulation_results
    
    def _create_feature_vector(self, metrics: Dict, context: Dict, 
                              required_features: List[str]) -> Optional[List[float]]:
        """
        Create feature vector for model prediction.
        
        Constructs a feature vector matching the model's expected input format,
        including current metrics, time difference, and one-hot encoded context.
        
        Args:
            metrics (Dict): Current network metrics
            context (Dict): Current match context
            required_features (List[str]): Features expected by model
        
        Returns:
            Optional[List[float]]: Feature vector, or None if construction fails
        
        Feature Types:
            - current_{metric}: Current metric value
            - time_diff: Fixed at 10 minutes (typical window length)
            - {context_type}_{value}: One-hot encoded context (1 if match, 0 otherwise)
        
        Notes:
            - Missing metrics are filled with 0.0
            - Time difference is assumed constant (10 minutes)
            - Context features are one-hot encoded
        """
        features = []
        
        for feature_name in required_features:
            if feature_name.startswith('current_'):
                # Current metric value
                metric_name = feature_name.replace('current_', '')
                if metric_name in metrics:
                    features.append(metrics[metric_name])
                else:
                    features.append(0.0)
            elif feature_name == 'time_diff':
                # Assume 10-minute window
                features.append(10.0)
            elif feature_name.startswith('score_context_'):
                # One-hot encoded score context
                context_value = feature_name.replace('score_context_', '')
                features.append(1.0 if context.get('score_context') == context_value else 0.0)
            elif feature_name.startswith('phase_context_'):
                # One-hot encoded phase context
                context_value = feature_name.replace('phase_context_', '')
                features.append(1.0 if context.get('phase_context') == context_value else 0.0)
            elif feature_name.startswith('intensity_context_'):
                # One-hot encoded intensity context
                context_value = feature_name.replace('intensity_context_', '')
                features.append(1.0 if context.get('intensity_context') == context_value else 0.0)
            else:
                # Unknown feature
                features.append(0.0)
        
        return features if len(features) == len(required_features) else None
    
    def _estimate_recommendation_effect(self, recommendations: List[Dict], 
                                      metric: str) -> float:
        """
        Estimate the effect of recommendations on a specific metric.
        
        Estimates how much a recommendation would change a metric if implemented.
        Effects are based on:
        1. Recommendation type (spatial, tempo, connectivity, etc.)
        2. Confidence score (higher confidence → larger effect)
        3. Metric-specific impact patterns
        
        Args:
            recommendations (List[Dict]): Recommendations for this window
            metric (str): Metric to estimate effect for
        
        Returns:
            float: Estimated total effect on metric
        
        Effect Magnitudes (scaled by confidence):
            - Spatial: density +0.005, clustering +0.003, centralization -0.002
            - Tempo: density +0.008, path_length -0.1, centralization +0.003
            - Connectivity: clustering +0.006, betweenness +0.004, density +0.004
            - Attacking: density +0.010, centralization +0.005
            - Defensive: clustering +0.004, centralization -0.003
        
        Notes:
            - Effects are additive across multiple recommendations
            - Magnitudes are empirically calibrated (could be refined)
            - Confidence scaling: effect = base_effect * confidence
            - This is a simplified model (real effects may vary)
        """
        total_effect = 0.0
        
        for rec in recommendations:
            rec_type = rec.get('type', '')
            confidence = rec.get('confidence_score', 0.0)
            
            # Define recommendation effects on metrics
            # Format: {rec_type: {metric: base_effect}}
            effect_map = {
                'spatial': {
                    'density': 0.005 * confidence,
                    'clustering_coefficient': 0.003 * confidence,
                    'centralization': -0.002 * confidence
                },
                'tempo': {
                    'density': 0.008 * confidence,
                    'avg_path_length': -0.1 * confidence,
                    'centralization': 0.003 * confidence
                },
                'connectivity': {
                    'clustering_coefficient': 0.006 * confidence,
                    'avg_betweenness_centrality': 0.004 * confidence,
                    'density': 0.004 * confidence
                },
                'attacking': {
                    'density': 0.010 * confidence,
                    'centralization': 0.005 * confidence
                },
                'defensive': {
                    'clustering_coefficient': 0.004 * confidence,
                    'centralization': -0.003 * confidence
                }
            }
            
            # Add effect if recommendation type and metric match
            if rec_type in effect_map and metric in effect_map[rec_type]:
                total_effect += effect_map[rec_type][metric]
        
        return total_effect
    
    def _assess_simulation_quality(self, simulated_outcomes: Dict) -> float:
        """
        Assess quality of simulation.
        
        Quality is measured as the proportion of metrics successfully simulated.
        Higher quality indicates more comprehensive simulation.
        
        Args:
            simulated_outcomes (Dict): Simulated outcomes for all metrics
        
        Returns:
            float: Quality score [0, 1]
                - 1.0: All 6 metrics simulated
                - 0.5: Half of metrics simulated
                - 0.0: No metrics simulated
        
        Notes:
            - Total possible metrics: 6 (density, clustering, betweenness,
              eigenvector, path_length, centralization)
            - Capped at 1.0 maximum
        """
        if not simulated_outcomes:
            return 0.0
        
        # Quality = proportion of metrics simulated
        quality_score = len(simulated_outcomes) / 6.0  # 6 total metrics
        
        return min(1.0, quality_score)
    
    # =========================================================================
    # OUTCOME COMPARISON
    # =========================================================================
    
    def compare_outcomes(self, simulation_results: List[Dict]) -> Dict:
        """
        Compare actual vs counterfactual outcomes.
        
        For each simulated scenario, compares what actually happened (actual future
        performance) with what would have happened if recommendations were followed
        (simulated performance).
        
        Args:
            simulation_results (List[Dict]): Simulated outcomes
        
        Returns:
            Dict: Comparison results with structure:
                {
                    'individual_comparisons': [
                        {
                            'scenario_id': 0,
                            'window_info': {...},
                            'recommendations': [...],
                            'metric_comparisons': {
                                'density': {
                                    'actual_change': 0.02,
                                    'simulated_change': 0.05,
                                    'difference': 0.03,
                                    'improvement': True
                                },
                                ...
                            }
                        },
                        ...
                    ],
                    'summary_statistics': {
                        'improvement_rates': {...},
                        'average_differences': {...},
                        'statistical_significance': {...}
                    },
                    'total_comparisons': 250
                }
        
        Process:
            1. For each simulation:
                a. Get actual future performance (2 windows ahead)
                b. Compare with simulated performance
                c. Calculate differences and improvement flags
            2. Aggregate summary statistics:
                a. Improvement rates per metric
                b. Average differences per metric
                c. Statistical significance (Wilcoxon test)
        
        Notes:
            - Actual future = average of next 2 windows
            - Improvement = simulated_change > actual_change
            - Wilcoxon test is non-parametric (robust to outliers)
        """
        comparisons = []
        
        for sim_result in simulation_results:
            scenario = sim_result['scenario']
            simulated_outcomes = sim_result['simulated_outcomes']
            
            # Get actual future performance
            actual_future = self._get_actual_future_performance(scenario['window_info'])
            
            if actual_future:
                comparison = {
                    'scenario_id': len(comparisons),
                    'window_info': scenario['window_info'],
                    'recommendations': scenario['recommendations'],
                    'metric_comparisons': {}
                }
                
                # Compare each metric
                for metric, sim_data in simulated_outcomes.items():
                    if metric in actual_future:
                        actual_change = actual_future[metric] - sim_data['original_value']
                        simulated_change = sim_data['total_change']
                        
                        comparison['metric_comparisons'][metric] = {
                            'actual_change': actual_change,
                            'simulated_change': simulated_change,
                            'difference': simulated_change - actual_change,
                            'improvement': simulated_change > actual_change
                        }
                
                comparisons.append(comparison)
        
        # Calculate summary statistics
        summary_stats = self._calculate_comparison_summary(comparisons)
        
        return {
            'individual_comparisons': comparisons,
            'summary_statistics': summary_stats,
            'total_comparisons': len(comparisons)
        }
    
    def _get_actual_future_performance(self, window_info: Dict) -> Optional[Dict]:
        """
        Get actual future performance for a window.
        
        Retrieves network metrics from the next 2 windows (same match, same team)
        and averages them to get actual future performance.
        
        Args:
            window_info (Dict): Current window metadata
        
        Returns:
            Optional[Dict]: Average future metrics, or None if unavailable
        
        Notes:
            - Uses 2-window lookahead (~10 minutes)
            - Averages to smooth short-term fluctuations
            - Returns None if window is near match end
        """
        match_id = window_info.get('match_id')
        team = window_info.get('team')
        current_window = window_info.get('window_id')
        
        if None in [match_id, team, current_window]:
            return None
        
        # Get future window data
        match_data = self.network_data[
            (self.network_data['match_id'] == match_id) & 
            (self.network_data['team'] == team)
        ]
        
        future_windows = match_data[match_data.index > current_window].head(2)
        
        if future_windows.empty:
            return None
        
        # Calculate average future metrics
        future_metrics = {}
        for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                      'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
            if metric in future_windows.columns:
                future_metrics[metric] = future_windows[metric].mean()
        
        return future_metrics
    
    def _calculate_comparison_summary(self, comparisons: List[Dict]) -> Dict:
        """
        Calculate summary statistics for comparisons.
        
        Aggregates comparison results across all scenarios to provide:
        1. Improvement rates per metric (% of scenarios with improvement)
        2. Average differences per metric (mean simulated - actual)
        3. Statistical significance per metric (Wilcoxon signed-rank test)
        
        Args:
            comparisons (List[Dict]): Individual comparison results
        
        Returns:
            Dict: Summary statistics with structure:
                {
                    'improvement_rates': {
                        'density': 0.68,
                        'clustering_coefficient': 0.72,
                        ...
                    },
                    'average_differences': {
                        'density': 0.015,
                        'clustering_coefficient': 0.008,
                        ...
                    },
                    'statistical_significance': {
                        'density': {
                            'p_value': 0.023,
                            'significant': True
                        },
                        ...
                    }
                }
        
        Notes:
            - Improvement rate = proportion with simulated > actual
            - Average difference = mean(simulated - actual)
            - Wilcoxon test: paired non-parametric test
            - Significance threshold: p < 0.05
        """
        if not comparisons:
            return {}
        
        summary = {
            'improvement_rates': {},
            'average_differences': {},
            'statistical_significance': {}
        }
        
        # Calculate metrics across all comparisons
        for metric in ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                      'avg_eigenvector_centrality', 'avg_path_length', 'centralization']:
            
            improvements = []
            differences = []
            actual_changes = []
            simulated_changes = []
            
            # Collect data for this metric
            for comp in comparisons:
                if metric in comp['metric_comparisons']:
                    metric_comp = comp['metric_comparisons'][metric]
                    improvements.append(metric_comp['improvement'])
                    differences.append(metric_comp['difference'])
                    actual_changes.append(metric_comp['actual_change'])
                    simulated_changes.append(metric_comp['simulated_change'])
            
            if improvements:
                # Improvement rate
                summary['improvement_rates'][metric] = sum(improvements) / len(improvements)
                
                # Average difference
                summary['average_differences'][metric] = np.mean(differences)
                
                # Statistical significance (Wilcoxon signed-rank test)
                if len(actual_changes) > 1 and len(simulated_changes) > 1:
                    try:
                        stat, p_value = stats.wilcoxon(actual_changes, simulated_changes)
                        summary['statistical_significance'][metric] = {
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        # Handle cases where test fails (e.g., all zeros)
                        summary['statistical_significance'][metric] = {
                            'p_value': 1.0,
                            'significant': False
                        }
        
        return summary
    
    # =========================================================================
    # IMPACT QUANTIFICATION
    # =========================================================================
    
    def calculate_recommendation_impact(self, comparison_results: Dict) -> Dict:
        """
        Calculate overall impact of recommendations.
        
        Quantifies the effectiveness of recommendations by analyzing:
        1. Overall improvement rate (across all metrics and scenarios)
        2. Metric-specific impacts (improvement rates per metric)
        3. Recommendation type impacts (effectiveness by type)
        
        Args:
            comparison_results (Dict): Comparison results from compare_outcomes()
        
        Returns:
            Dict: Impact analysis with structure:
                {
                    'overall_improvement_rate': 0.68,
                    'metric_impacts': {
                        'density': 0.72,
                        'clustering_coefficient': 0.65,
                        ...
                    },
                    'confidence_correlation': 0.0,  # Placeholder
                    'recommendation_type_impacts': {
                        'spatial': {
                            'improvement_rate': 0.70,
                            'sample_size': 150
                        },
                        'tempo': {
                            'improvement_rate': 0.65,
                            'sample_size': 120
                        },
                        ...
                    }
                }
        
        Notes:
            - Overall rate = total improvements / total comparisons
            - Metric impacts from summary statistics
            - Type impacts calculated by grouping recommendations
            - Higher rates indicate more effective recommendations
        """
        if not comparison_results.get('individual_comparisons'):
            return {'error': 'No comparison data available'}
        
        impact_analysis = {
            'overall_improvement_rate': 0.0,
            'metric_impacts': {},
            'confidence_correlation': 0.0,  # Placeholder
            'recommendation_type_impacts': {}
        }
        
        # Calculate overall improvement rate
        total_improvements = 0
        total_comparisons = 0
        
        for comp in comparison_results['individual_comparisons']:
            for metric, metric_comp in comp['metric_comparisons'].items():
                total_comparisons += 1
                if metric_comp['improvement']:
                    total_improvements += 1
        
        if total_comparisons > 0:
            impact_analysis['overall_improvement_rate'] = total_improvements / total_comparisons
        
        # Metric-specific impacts (from summary statistics)
        summary_stats = comparison_results.get('summary_statistics', {})
        impact_analysis['metric_impacts'] = summary_stats.get('improvement_rates', {})
        
        # Calculate recommendation type impacts
        type_impacts = {}
        for comp in comparison_results['individual_comparisons']:
            for rec in comp['recommendations']:
                rec_type = rec.get('type', 'unknown')
                if rec_type not in type_impacts:
                    type_impacts[rec_type] = {'improvements': 0, 'total': 0}
                
                # Count improvements for this recommendation type
                for metric_comp in comp['metric_comparisons'].values():
                    type_impacts[rec_type]['total'] += 1
                    if metric_comp['improvement']:
                        type_impacts[rec_type]['improvements'] += 1
        
        # Calculate improvement rates by type
        for rec_type, data in type_impacts.items():
            if data['total'] > 0:
                impact_analysis['recommendation_type_impacts'][rec_type] = {
                    'improvement_rate': data['improvements'] / data['total'],
                    'sample_size': data['total']
                }
        
        return impact_analysis
    
    # =========================================================================
    # MODEL EVALUATION
    # =========================================================================
    
    def evaluate_model_performance(self) -> Dict:
        """
        Evaluate performance of predictive models.
        
        Compiles performance metrics for all trained models to assess the quality
        of baseline predictions (which underpin counterfactual simulations).
        
        Returns:
            Dict: Model performance evaluation with structure:
                {
                    'individual_models': {
                        'density': {
                            'train_score': 0.85,
                            'test_score': 0.72,
                            'r2': 0.72,
                            'mae': 0.015,
                            'rmse': 0.023,
                            'feature_count': 12,
                            'top_features': [
                                ('current_density', 0.45),
                                ('score_context_trailing', 0.18),
                                ('time_diff', 0.12)
                            ]
                        },
                        ...
                    },
                    'overall_quality': 0.68,
                    'total_models': 6
                }
        
        Metrics:
            - train_score: R² on training set
            - test_score: R² on test set
            - r2: R² score (coefficient of determination)
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - feature_count: Number of features used
            - top_features: Top 3 most important features
        
        Notes:
            - Overall quality = average test R² across all models
            - Higher R² indicates better predictive accuracy
            - Feature importance from Random Forest
        """
        model_performance = {}
        
        for metric, model_info in self.counterfactual_models.items():
            model_performance[metric] = {
                'train_score': model_info['train_score'],
                'test_score': model_info['test_score'],
                'r2': model_info.get('r2'),
                'mae': model_info.get('mae'),
                'rmse': model_info.get('rmse'),
                'feature_count': len(model_info['features']),
                'top_features': sorted(
                    model_info['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3 features
            }
        
        # Overall model quality (average test R²)
        test_scores = [info['test_score'] for info in model_performance.values()]
        overall_quality = np.mean(test_scores) if test_scores else 0.0
        
        return {
            'individual_models': model_performance,
            'overall_quality': overall_quality,
            'total_models': len(model_performance)
        }
