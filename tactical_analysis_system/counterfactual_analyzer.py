import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

class CounterfactualAnalyzer:
    """Analyze counterfactual scenarios for tactical recommendations"""
    
    def __init__(self, network_data: pd.DataFrame, recommendations_data: List[Dict]):
        self.network_data = network_data
        self.recommendations_data = recommendations_data
        self.counterfactual_models = {}
        self.simulation_results = {}
        
    def run_counterfactual_analysis(self) -> Dict:
        """Run comprehensive counterfactual analysis"""
        
        print("Running Counterfactual Analysis...")
        print("=" * 50)
        
        # 1. Build predictive models
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
    
    def build_predictive_models(self):
        """Build models to predict network metric changes"""
        
        # Prepare training data
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
                    print(f"   ✓ Built model for {metric}")
    
    def _prepare_training_data(self) -> pd.DataFrame:
        """Prepare training data for predictive models"""
        
        training_rows = []
        
        # Create training examples from consecutive windows
        for match_id in self.network_data['match_id'].unique():
            match_data = self.network_data[
                self.network_data['match_id'] == match_id
            ].sort_values('start_minute')
            
            for i in range(len(match_data) - 1):
                current_window = match_data.iloc[i]
                next_window = match_data.iloc[i + 1]
                
                # Calculate changes
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
                
                # Target changes
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
        """Build predictive model for a specific metric"""
        
        target_col = f'{metric}_change'
        
        # Select features
        feature_cols = [
            f'current_{metric}', 'time_diff'
        ]
        
        # Add context features (encoded)
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        return {
            'model': model,
            'scaler': scaler,
            'features': available_features,
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': dict(zip(available_features, model.feature_importances_))
        }
    
    def identify_counterfactual_scenarios(self) -> List[Dict]:
        """Identify scenarios for counterfactual analysis"""
        
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
    
    def simulate_alternative_outcomes(self, scenarios: List[Dict]) -> List[Dict]:
        """Simulate outcomes if recommendations were followed"""
        
        simulation_results = []
        
        for scenario in scenarios:
            # Get actual metrics
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
                        # Predict change
                        features_scaled = model_info['scaler'].transform([features])
                        predicted_change = model_info['model'].predict(features_scaled)[0]
                        
                        # Apply recommendation effect (simplified)
                        recommendation_effect = self._estimate_recommendation_effect(
                            recommendations, metric
                        )
                        
                        # Combine predicted change with recommendation effect
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
        """Create feature vector for model prediction"""
        
        features = []
        
        for feature_name in required_features:
            if feature_name.startswith('current_'):
                metric_name = feature_name.replace('current_', '')
                if metric_name in metrics:
                    features.append(metrics[metric_name])
                else:
                    features.append(0.0)
            elif feature_name == 'time_diff':
                features.append(10.0)  # Assume 10-minute window
            elif feature_name.startswith('score_context_'):
                context_value = feature_name.replace('score_context_', '')
                features.append(1.0 if context.get('score_context') == context_value else 0.0)
            elif feature_name.startswith('phase_context_'):
                context_value = feature_name.replace('phase_context_', '')
                features.append(1.0 if context.get('phase_context') == context_value else 0.0)
            elif feature_name.startswith('intensity_context_'):
                context_value = feature_name.replace('intensity_context_', '')
                features.append(1.0 if context.get('intensity_context') == context_value else 0.0)
            else:
                features.append(0.0)
        
        return features if len(features) == len(required_features) else None
    
    def _estimate_recommendation_effect(self, recommendations: List[Dict], 
                                      metric: str) -> float:
        """Estimate the effect of recommendations on a specific metric"""
        
        # Simplified recommendation effect estimation
        # In practice, this would be more sophisticated
        
        total_effect = 0.0
        
        for rec in recommendations:
            rec_type = rec.get('type', '')
            confidence = rec.get('confidence_score', 0.0)
            
            # Define recommendation effects on metrics
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
            
            if rec_type in effect_map and metric in effect_map[rec_type]:
                total_effect += effect_map[rec_type][metric]
        
        return total_effect
    
    def _assess_simulation_quality(self, simulated_outcomes: Dict) -> float:
        """Assess quality of simulation"""
        
        if not simulated_outcomes:
            return 0.0
        
        # Simple quality assessment based on number of metrics simulated
        quality_score = len(simulated_outcomes) / 6.0  # 6 total metrics
        
        return min(1.0, quality_score)
    
    def compare_outcomes(self, simulation_results: List[Dict]) -> Dict:
        """Compare actual vs counterfactual outcomes"""
        
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
        """Get actual future performance for a window"""
        
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
        """Calculate summary statistics for comparisons"""
        
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
            
            for comp in comparisons:
                if metric in comp['metric_comparisons']:
                    metric_comp = comp['metric_comparisons'][metric]
                    improvements.append(metric_comp['improvement'])
                    differences.append(metric_comp['difference'])
                    actual_changes.append(metric_comp['actual_change'])
                    simulated_changes.append(metric_comp['simulated_change'])
            
            if improvements:
                summary['improvement_rates'][metric] = sum(improvements) / len(improvements)
                summary['average_differences'][metric] = np.mean(differences)
                
                # Statistical test
                if len(actual_changes) > 1 and len(simulated_changes) > 1:
                    stat, p_value = stats.wilcoxon(actual_changes, simulated_changes)
                    summary['statistical_significance'][metric] = {
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        return summary
    
    def calculate_recommendation_impact(self, comparison_results: Dict) -> Dict:
        """Calculate overall impact of recommendations"""
        
        if not comparison_results.get('individual_comparisons'):
            return {'error': 'No comparison data available'}
        
        impact_analysis = {
            'overall_improvement_rate': 0.0,
            'metric_impacts': {},
            'confidence_correlation': 0.0,
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
        
        # Calculate metric-specific impacts
        summary_stats = comparison_results.get('summary_statistics', {})
        impact_analysis['metric_impacts'] = summary_stats.get('improvement_rates', {})
        
        # Calculate recommendation type impacts
        type_impacts = {}
        for comp in comparison_results['individual_comparisons']:
            for rec in comp['recommendations']:
                rec_type = rec.get('type', 'unknown')
                if rec_type not in type_impacts:
                    type_impacts[rec_type] = {'improvements': 0, 'total': 0}
                
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
    
    def evaluate_model_performance(self) -> Dict:
        """Evaluate performance of predictive models"""
        
        model_performance = {}
        
        for metric, model_info in self.counterfactual_models.items():
            model_performance[metric] = {
                'train_score': model_info['train_score'],
                'test_score': model_info['test_score'],
                'feature_count': len(model_info['features']),
                'top_features': sorted(
                    model_info['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
        
        # Overall model quality
        test_scores = [info['test_score'] for info in model_performance.values()]
        overall_quality = np.mean(test_scores) if test_scores else 0.0
        
        return {
            'individual_models': model_performance,
            'overall_quality': overall_quality,
            'total_models': len(model_performance)
        }
