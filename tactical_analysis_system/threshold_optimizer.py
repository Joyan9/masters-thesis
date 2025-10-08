"""
FIXED Threshold Optimization Module
Addresses data format issues, constraint validation, and objective function robustness
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from .tactical_recommender import TacticalRecommender, ThresholdAnalyzer, RuleEngine
from .recommendation_validator import RecommendationValidator


class ImprovedThresholdOptimizer:
    """
    FIXED: Robust threshold optimization with proper error handling
    """
    
    def __init__(self, 
                 network_data: pd.DataFrame,
                 validation_weights: Optional[Dict[str, float]] = None):
        """
        Initialize optimizer with simplified, robust approach
        
        Args:
            network_data: Network metrics DataFrame from RQ1
            validation_weights: Weights for validation components
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("Install: pip install scikit-optimize")
        
        self.network_data = network_data.copy()
        self.validation_weights = validation_weights or {
            'performance_outcomes': 0.35,
            'temporal_consistency': 0.20,
            'context_sensitivity': 0.20,
            'recommendation_effectiveness': 0.15,
            'elite_pattern_alignment': 0.10
        }
        
        # Validate data
        self._validate_input_data()
        
        # Extract metric statistics for threshold bounds
        self.metric_stats = self._compute_metric_statistics()
        
        # Create search space (optimize multipliers, not percentiles)
        self.search_space = self._create_search_space()
        
        # Optimization state
        self.optimization_history = []
        self.best_thresholds = None
        self.best_score = -np.inf
        self.iteration_count = 0
        
        print(f"✅ ImprovedThresholdOptimizer initialized")
        print(f"   Data: {len(self.network_data)} windows, {self.network_data['match_id'].nunique()} matches")
        print(f"   Metrics: {list(self.metric_stats.keys())}")
    
    def _validate_input_data(self):
        """Validate input data has required columns"""
        required_cols = ['match_id', 'team']
        metric_cols = ['density', 'clustering_coefficient', 'centralization', 'avg_path_length']
        
        missing = [col for col in required_cols if col not in self.network_data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        available_metrics = [col for col in metric_cols if col in self.network_data.columns]
        if len(available_metrics) < 2:
            raise ValueError(f"Need at least 2 network metrics. Found: {available_metrics}")
        
        print(f"   Available metrics: {available_metrics}")
    
    def _compute_metric_statistics(self) -> Dict:
        """Compute statistics for each metric to define bounds"""
        stats = {}
        
        metrics = ['density', 'clustering_coefficient', 'centralization', 
                  'avg_path_length', 'avg_betweenness_centrality', 'avg_eigenvector_centrality']
        
        for metric in metrics:
            if metric in self.network_data.columns:
                values = self.network_data[metric].dropna()
                if len(values) > 10:  # Need sufficient data
                    stats[metric] = {
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'p10': float(np.percentile(values, 10)),
                        'p25': float(np.percentile(values, 25)),
                        'p50': float(np.percentile(values, 50)),
                        'p75': float(np.percentile(values, 75)),
                        'p90': float(np.percentile(values, 90))
                    }
        
        return stats
    
    def _create_search_space(self) -> List:
        """
        Create search space using MULTIPLIERS instead of percentiles
        
        This is more robust: we optimize how much to shift from baseline percentiles
        """
        search_space = []
        
        for metric in self.metric_stats.keys():
            # Optimize multipliers for threshold shifts
            # excellent_shift: how much to shift the 90th percentile threshold
            search_space.append(Real(-0.3, 0.3, name=f'{metric}_excellent_shift'))
            search_space.append(Real(-0.3, 0.3, name=f'{metric}_good_shift'))
            search_space.append(Real(-0.3, 0.3, name=f'{metric}_poor_shift'))
            search_space.append(Real(-0.3, 0.3, name=f'{metric}_critical_shift'))
        
        return search_space
    
    def _shifts_to_thresholds(self, shifts: Dict[str, float]) -> Dict:
        """
        Convert shift multipliers to actual threshold values
        FIXED: Proper handling of inverted metrics
        """
        thresholds = {}
        
        for metric, stats in self.metric_stats.items():
            # Base thresholds (from percentiles)
            base_excellent = stats['p90']
            base_good = stats['p75']
            base_average = stats['p50']
            base_poor = stats['p25']
            base_critical = stats['p10']
            
            # Apply shifts (as fraction of std)
            std = stats['std']
            
            excellent_shift = shifts.get(f'{metric}_excellent_shift', 0.0)
            good_shift = shifts.get(f'{metric}_good_shift', 0.0)
            poor_shift = shifts.get(f'{metric}_poor_shift', 0.0)
            critical_shift = shifts.get(f'{metric}_critical_shift', 0.0)
            
            # Calculate shifted thresholds
            excellent = base_excellent + (excellent_shift * std)
            good = base_good + (good_shift * std)
            average = base_average  # Keep average fixed at median
            poor = base_poor + (poor_shift * std)
            critical = base_critical + (critical_shift * std)
            
            # FIXED: Enforce ordering constraints BEFORE clipping
            if metric == 'avg_path_length':
                # Inverted: lower is better
                # Order should be: critical >= poor >= average >= good >= excellent
                # Which means: excellent <= good <= average <= poor <= critical
                
                # Ensure excellent is the lowest (best)
                excellent = min(excellent, average - 0.01 * std)  # Force below average
                
                # Ensure good is between excellent and average
                good = np.clip(good, excellent + 0.005 * std, average - 0.005 * std)
                
                # Ensure poor is between average and critical
                poor = np.clip(poor, average + 0.005 * std, critical - 0.005 * std)
                
                # Ensure critical is the highest (worst)
                critical = max(critical, average + 0.01 * std)  # Force above average
                
            else:
                # Normal: higher is better
                # Order should be: excellent >= good >= average >= poor >= critical
                
                # Ensure excellent is the highest (best)
                excellent = max(excellent, average + 0.01 * std)  # Force above average
                
                # Ensure good is between average and excellent
                good = np.clip(good, average + 0.005 * std, excellent - 0.005 * std)
                
                # Ensure poor is between critical and average
                poor = np.clip(poor, critical + 0.005 * std, average - 0.005 * std)
                
                # Ensure critical is the lowest (worst)
                critical = min(critical, average - 0.01 * std)  # Force below average
            
            # Clamp to data range (with small buffer)
            data_min = stats['min']
            data_max = stats['max']
            data_range = data_max - data_min
            buffer = 0.01 * data_range  # 1% buffer
            
            excellent = np.clip(excellent, data_min - buffer, data_max + buffer)
            good = np.clip(good, data_min - buffer, data_max + buffer)
            average = np.clip(average, data_min - buffer, data_max + buffer)
            poor = np.clip(poor, data_min - buffer, data_max + buffer)
            critical = np.clip(critical, data_min - buffer, data_max + buffer)
            
            # Final ordering check and correction
            if metric == 'avg_path_length':
                # Ensure: excellent <= good <= average <= poor <= critical
                values = sorted([excellent, good, average, poor, critical])
                excellent, good, average, poor, critical = values
            else:
                # Ensure: critical <= poor <= average <= good <= excellent
                values = sorted([critical, poor, average, good, excellent])
                critical, poor, average, good, excellent = values
            
            thresholds[metric] = {
                'excellent': float(excellent),
                'good': float(good),
                'average': float(average),
                'poor': float(poor),
                'critical': float(critical)
            }
        
        return thresholds

    def _validate_thresholds(self, thresholds: Dict) -> bool:
        """Strict validation of threshold ordering"""
        for metric, levels in thresholds.items():
            e = levels['excellent']
            g = levels['good']
            a = levels['average']
            p = levels['poor']
            c = levels['critical']
            
            if metric == 'avg_path_length':
                # Inverted
                if not (c >= p >= a >= g >= e):
                    return False
            else:
                # Normal
                if not (e >= g >= a >= p >= c):
                    return False
            
            # Check for degenerate cases
            if e == c:  # All thresholds collapsed
                return False
        
        return True
    
    def _compute_validation_score(self, shift_values: np.ndarray) -> float:
        """
        FIXED objective function with proper error handling
        """
        self.iteration_count += 1
        
        try:
            # Convert shifts to thresholds
            param_names = [space.name for space in self.search_space]
            shifts = {name: float(val) for name, val in zip(param_names, shift_values)}
            
            thresholds = self._shifts_to_thresholds(shifts)
            
            # Validate thresholds
            if not self._validate_thresholds(thresholds):
                if self.iteration_count <= 5:
                    print(f"   Iter {self.iteration_count}: Invalid threshold ordering, skipping")
                return -1.0
            
            # Create recommender with these thresholds
            recommender = TacticalRecommender()
            recommender.threshold_analyzer.thresholds = thresholds
            
            # CRITICAL FIX: Initialize rule engine properly
            recommender.rule_engine = RuleEngine(thresholds)
            
            # Generate recommendations - FIXED data structure
            all_recommendations = []
            
            for match_id in self.network_data['match_id'].unique():
                match_data = self.network_data[
                    self.network_data['match_id'] == match_id
                ].copy()
                
                if len(match_data) < 3:  # Skip matches with too few windows
                    continue
                
                # Generate match recommendations
                match_result = recommender.analyze_match_recommendations(
                    match_data, 
                    match_id=str(match_id)
                )
                
                # CRITICAL FIX: Append the full match result
                all_recommendations.append(match_result)
            
            if len(all_recommendations) == 0:
                if self.iteration_count <= 5:
                    print(f"   Iter {self.iteration_count}: No recommendations generated")
                return -0.5
            
            # Run validation - FIXED: pass correct data structure
            validator = RecommendationValidator(
                self.network_data,
                all_recommendations  # This is now List[Dict] with 'window_recommendations'
            )
            
            validation_results = validator.run_comprehensive_validation()
            
            # Extract scores
            overall_score_data = validation_results.get('overall_validation_score', {})
            component_scores = overall_score_data.get('component_scores', {})
            
            # Calculate weighted score
            weighted_score = 0.0
            for component, weight in self.validation_weights.items():
                score = component_scores.get(component, 0.0)
                if not np.isnan(score):
                    weighted_score += weight * score
            
            # Activation rate check
            total_windows = sum(
                len(m.get('window_recommendations', [])) 
                for m in all_recommendations
            )
            total_with_recs = sum(
                sum(1 for w in m.get('window_recommendations', []) 
                    if len(w.get('recommendations', [])) > 0)
                for m in all_recommendations
            )
            
            activation_rate = total_with_recs / total_windows if total_windows > 0 else 0
            
            # Penalty for extreme activation rates
            if activation_rate < 0.05 or activation_rate > 0.70:
                penalty = 0.2
            elif activation_rate < 0.10 or activation_rate > 0.60:
                penalty = 0.1
            else:
                penalty = 0.0
            
            final_score = weighted_score - penalty
            
            # Track history
            self.optimization_history.append({
                'iteration': self.iteration_count,
                'shifts': shifts,
                'thresholds': thresholds,
                'validation_score': weighted_score,
                'activation_rate': activation_rate,
                'activation_penalty': penalty,
                'final_score': final_score,
                'component_scores': component_scores
            })
            
            # Update best
            if final_score > self.best_score:
                self.best_score = final_score
                self.best_thresholds = thresholds
                print(f"🎯 Iter {self.iteration_count}: New best = {final_score:.4f} "
                      f"(val: {weighted_score:.4f}, act: {activation_rate:.1%}, pen: {penalty:.3f})")
            elif self.iteration_count % 20 == 0:
                print(f"   Iter {self.iteration_count}: score = {final_score:.4f}, "
                      f"best = {self.best_score:.4f}")
            
            return final_score
            
        except Exception as e:
            if self.iteration_count <= 5:
                print(f"⚠️  Iter {self.iteration_count} error: {str(e)[:100]}")
            return -1.0
    
    def optimize(self, 
             n_iterations: int = 100,
             n_initial_points: int = 20,
             random_state: int = 42,
             run_diagnostics: bool = True) -> Dict:  # ← Add this parameter
        """Run optimization with optional diagnostics"""
        
        print("\n" + "="*60)
        print("IMPROVED THRESHOLD OPTIMIZATION")
        print("="*60)
        print(f"Iterations: {n_iterations}")
        print(f"Search space: {len(self.search_space)} parameters")
        print()
        
        # RUN DIAGNOSTICS FIRST
        if run_diagnostics:
            baseline_thresholds = self.diagnose_threshold_generation()
            
            # Ask user if they want to continue
            print("\n" + "="*60)
            response = input("Continue with optimization? (y/n): ")
            if response.lower() != 'y':
                return {'error': 'user_cancelled', 'diagnostics': baseline_thresholds}

        
        # Compute baseline
        print("Computing baseline...")
        baseline_shifts = np.zeros(len(self.search_space))
        baseline_score = self._compute_validation_score(baseline_shifts)
        print(f"✅ Baseline score: {baseline_score:.4f}\n")
        
        if baseline_score < 0:
            print("❌ ERROR: Baseline validation failed!")
            print("   Check that your data has valid recommendations")
            return {
                'error': 'baseline_failed',
                'baseline_score': baseline_score
            }
        
        # Reset for optimization
        self.iteration_count = 0
        self.best_score = baseline_score
        self.optimization_history = []
        
        # Define objective
        @use_named_args(self.search_space)
        def objective(**params):
            shift_values = np.array([params[space.name] for space in self.search_space])
            return -self._compute_validation_score(shift_values)  # Negate for minimization
        
        # Run optimization
        print("Starting optimization...")
        print("-" * 60)
        
        result = gp_minimize(
            objective,
            self.search_space,
            n_calls=n_iterations,
            n_initial_points=n_initial_points,
            random_state=random_state,
            verbose=False
        )
        
        print("-" * 60)
        print(f"\n✅ Optimization complete!")
        print(f"   Best score: {self.best_score:.4f}")
        print(f"   Improvement: {self.best_score - baseline_score:+.4f}")
        
        return {
            'best_thresholds': self.best_thresholds,
            'best_score': self.best_score,
            'baseline_score': baseline_score,
            'improvement': self.best_score - baseline_score,
            'total_iterations': self.iteration_count,
            'optimization_history': self.optimization_history,
            'metric_stats': self.metric_stats
        }
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save results with better formatting"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save thresholds
        thresholds_file = output_path / f"optimal_thresholds_{timestamp}.json"
        with open(thresholds_file, 'w') as f:
            json.dump(results['best_thresholds'], f, indent=2)
        
        # Save history
        history_file = output_path / f"optimization_history_{timestamp}.csv"
        history_df = pd.DataFrame([
            {
                'iteration': h['iteration'],
                'final_score': h['final_score'],
                'validation_score': h['validation_score'],
                'activation_rate': h['activation_rate'],
                'activation_penalty': h['activation_penalty'],
                **h['component_scores']
            }
            for h in results['optimization_history']
        ])
        history_df.to_csv(history_file, index=False)
        
        # Save report
        report_file = output_path / f"optimization_report_{timestamp}.txt"
        report = self._generate_report(results)
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n📁 Results saved to {output_path}/")
        
        return {
            'thresholds_file': str(thresholds_file),
            'history_file': str(history_file),
            'report_file': str(report_file)
        }
    
    def _generate_report(self, results: Dict) -> str:
        """Generate improved report"""
        lines = [
            "THRESHOLD OPTIMIZATION REPORT",
            "=" * 60,
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Iterations: {results['total_iterations']}",
            "",
            "RESULTS:",
            f"  Baseline:  {results['baseline_score']:.4f}",
            f"  Optimized: {results['best_score']:.4f}",
            f"  Improvement: {results['improvement']:+.4f} ({results['improvement']/results['baseline_score']*100:+.1f}%)",
            "",
            "OPTIMAL THRESHOLDS:",
            ""
        ]
        
        for metric, levels in results['best_thresholds'].items():
            lines.append(f"  {metric}:")
            for level in ['excellent', 'good', 'average', 'poor', 'critical']:
                if level in levels:
                    lines.append(f"    {level:10s}: {levels[level]:.4f}")
            lines.append("")
        
        # Add best iteration info
        if results['optimization_history']:
            best = max(results['optimization_history'], key=lambda x: x['final_score'])
            lines.extend([
                "BEST ITERATION:",
                f"  Iteration: {best['iteration']}",
                f"  Score: {best['final_score']:.4f}",
                f"  Activation rate: {best['activation_rate']:.1%}",
                "  Component scores:",
            ])
            for comp, score in best['component_scores'].items():
                lines.append(f"    {comp}: {score:.4f}")
        
        return "\n".join(lines)
 
    def plot_optimization_history(self, results: Dict, save_path: Optional[str] = None):
        """Plot optimization convergence"""
        try:
            import matplotlib.pyplot as plt
            
            history = results['optimization_history']
            iterations = [h['iteration'] for h in history]
            scores = [h['final_score'] for h in history]
            
            # Running best
            running_best = []
            current_best = -np.inf
            for score in scores:
                current_best = max(current_best, score)
                running_best.append(current_best)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iterations, scores, 'o-', alpha=0.5, label='Iteration Score')
            ax.plot(iterations, running_best, 'r-', linewidth=2, label='Best Score')
            ax.axhline(results['baseline_score'], color='g', linestyle='--', 
                      label=f'Baseline ({results["baseline_score"]:.3f})')
            
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Validation Score')
            ax.set_title('Threshold Optimization Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except ImportError:
            print("⚠️  matplotlib not available for plotting")


    def diagnose_threshold_generation(self):
        """
        Diagnostic function to see what's happening with threshold generation
        """
        print("\n" + "="*60)
        print("DIAGNOSTIC: THRESHOLD GENERATION")
        print("="*60)
        
        # 1. Check metric statistics
        print("\n1. METRIC STATISTICS:")
        print("-" * 60)
        for metric, stats in self.metric_stats.items():
            print(f"\n{metric}:")
            print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            print(f"  Percentiles:")
            print(f"    P10: {stats['p10']:.4f}")
            print(f"    P25: {stats['p25']:.4f}")
            print(f"    P50: {stats['p50']:.4f}")
            print(f"    P75: {stats['p75']:.4f}")
            print(f"    P90: {stats['p90']:.4f}")
        
        # 2. Test baseline threshold generation
        print("\n2. BASELINE THRESHOLD GENERATION (zero shifts):")
        print("-" * 60)
        
        baseline_shifts = {f'{metric}_{level}_shift': 0.0 
                        for metric in self.metric_stats.keys() 
                        for level in ['excellent', 'good', 'poor', 'critical']}
        
        baseline_thresholds = self._shifts_to_thresholds(baseline_shifts)
        
        for metric, levels in baseline_thresholds.items():
            print(f"\n{metric}:")
            for level in ['excellent', 'good', 'average', 'poor', 'critical']:
                print(f"  {level:10s}: {levels[level]:.6f}")
            
            # Check ordering
            e, g, a, p, c = levels['excellent'], levels['good'], levels['average'], levels['poor'], levels['critical']
            
            if metric == 'avg_path_length':
                valid = c >= p >= a >= g >= e
                expected_order = "critical >= poor >= average >= good >= excellent"
            else:
                valid = e >= g >= a >= p >= c
                expected_order = "excellent >= good >= average >= poor >= critical"
            
            print(f"  Expected order: {expected_order}")
            print(f"  Valid: {'✅' if valid else '❌'}")
            
            if not valid:
                print(f"  PROBLEM: Ordering violated!")
                if metric == 'avg_path_length':
                    if c < p: print(f"    critical ({c:.4f}) < poor ({p:.4f})")
                    if p < a: print(f"    poor ({p:.4f}) < average ({a:.4f})")
                    if a < g: print(f"    average ({a:.4f}) < good ({g:.4f})")
                    if g < e: print(f"    good ({g:.4f}) < excellent ({e:.4f})")
                else:
                    if e < g: print(f"    excellent ({e:.4f}) < good ({g:.4f})")
                    if g < a: print(f"    good ({g:.4f}) < average ({a:.4f})")
                    if a < p: print(f"    average ({a:.4f}) < poor ({p:.4f})")
                    if p < c: print(f"    poor ({p:.4f}) < critical ({c:.4f})")
        
        # 3. Check if validation passes
        print("\n3. VALIDATION CHECK:")
        print("-" * 60)
        is_valid = self._validate_thresholds(baseline_thresholds)
        print(f"Baseline thresholds valid: {'✅' if is_valid else '❌'}")
        
        # 4. Check data distribution
        print("\n4. DATA DISTRIBUTION CHECK:")
        print("-" * 60)
        for metric in ['density', 'clustering_coefficient', 'centralization', 'avg_path_length']:
            if metric in self.network_data.columns:
                values = self.network_data[metric].dropna()
                print(f"\n{metric}:")
                print(f"  N values: {len(values)}")
                print(f"  N unique: {values.nunique()}")
                print(f"  N zeros: {(values == 0).sum()}")
                print(f"  N negative: {(values < 0).sum()}")
                print(f"  Sample values: {values.head(10).tolist()}")
        
        return baseline_thresholds


def optimize_thresholds_standalone(network_data: pd.DataFrame,
                                   n_iterations: int = 100,
                                   output_dir: str = "results") -> Dict:
    """
    Standalone function to optimize thresholds
    
    Args:
        network_data: Network metrics DataFrame
        n_iterations: Number of optimization iterations
        output_dir: Directory to save results
        
    Returns:
        Optimization results dictionary
    """
    optimizer = ThresholdOptimizer(network_data)
    results = optimizer.optimize(n_iterations=n_iterations)
    optimizer.save_results(results, output_dir=output_dir)
    
    # Try to plot
    try:
        plot_path = Path(output_dir) / "optimization_convergence.png"
        optimizer.plot_optimization_history(results, save_path=str(plot_path))
    except:
        pass
    
    return results
