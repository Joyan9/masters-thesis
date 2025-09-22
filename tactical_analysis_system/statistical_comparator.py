import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro, kruskal, mannwhitneyu, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Dict, List, Tuple, Any

class StatisticalComparator:
    """Performs rigorous statistical comparisons between contexts"""
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.comparison_results = {}
    
    def compare_contexts(self, results_df: pd.DataFrame) -> Dict:
        """Compare network metrics across different contexts with proper statistical tests"""
        comparisons = {}
        
        # Define context types to analyze
        context_types = ['score_context', 'phase_context', 'intensity_context']
        
        for context_type in context_types:
            if context_type in results_df.columns:
                comparisons[context_type] = self._analyze_context_type(results_df, context_type)
        
        self.comparison_results = comparisons
        return comparisons
    
    def _analyze_context_type(self, data: pd.DataFrame, context_type: str) -> Dict:
        """Analyze a specific context type"""
        metrics = ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                  'avg_eigenvector_centrality', 'avg_path_length', 'centralization']
        
        results = {
            'context_type': context_type,
            'descriptive_stats': {},
            'normality_tests': {},
            'group_comparisons': {},
            'pairwise_comparisons': {},
            'effect_sizes': {}
        }
        
        # Get unique context labels
        context_labels = data[context_type].unique()
        
        # Descriptive statistics
        for metric in metrics:
            if metric in data.columns:
                desc_stats = data.groupby(context_type)[metric].agg([
                    'count', 'mean', 'std', 'median', 'min', 'max'
                ]).round(4)
                results['descriptive_stats'][metric] = desc_stats.to_dict('index')
        
        # Test each metric
        for metric in metrics:
            if metric in data.columns:
                metric_data = data[metric].dropna()
                if len(metric_data) > 0:
                    results['normality_tests'][metric] = self._test_normality(data, metric, context_type)
                    results['group_comparisons'][metric] = self._compare_groups(data, metric, context_type)
                    results['pairwise_comparisons'][metric] = self._pairwise_comparisons(data, metric, context_type)
                    results['effect_sizes'][metric] = self._calculate_effect_sizes(data, metric, context_type)
        
        return results
    
    def _test_normality(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """Test normality for each group"""
        normality_results = {}
        
        for context_label in data[context_type].unique():
            group_data = data[data[context_type] == context_label][metric].dropna()
            
            if len(group_data) >= 3:  # Minimum for Shapiro-Wilk
                try:
                    stat, p_value = shapiro(group_data)
                    normality_results[context_label] = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'is_normal': bool(p_value > self.alpha),
                        'n': int(len(group_data))
                    }
                except:
                    normality_results[context_label] = {
                        'statistic': None,
                        'p_value': None,
                        'is_normal': False,
                        'n': int(len(group_data))
                    }
            else:
                normality_results[context_label] = {
                    'statistic': None,
                    'p_value': None,
                    'is_normal': False,
                    'n': int(len(group_data))
                }
        
        # Overall normality assessment
        all_normal = all(result.get('is_normal', False) for result in normality_results.values())
        
        return {
            'by_group': normality_results,
            'all_groups_normal': all_normal,
            'recommended_test': 'parametric' if all_normal else 'non_parametric'
        }
    
    def _compare_groups(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """Compare groups using appropriate statistical test"""
        groups = []
        group_names = []
        
        for context_label in data[context_type].unique():
            group_data = data[data[context_type] == context_label][metric].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(context_label)
        
        if len(groups) < 2:
            return {'error': 'Insufficient groups for comparison'}
        
        # Check normality
        normality_result = self._test_normality(data, metric, context_type)
        use_parametric = normality_result['all_groups_normal']
        
        try:
            if use_parametric:
                # One-way ANOVA
                statistic, p_value = f_oneway(*groups)
                test_name = 'One-way ANOVA'
            else:
                # Kruskal-Wallis test
                statistic, p_value = kruskal(*groups)
                test_name = 'Kruskal-Wallis'
            
            return {
                'test_name': test_name,
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(p_value < self.alpha),
                'groups_compared': group_names,
                'n_groups': len(groups)
            }
        
        except Exception as e:
            return {'error': f'Statistical test failed: {str(e)}'}
    
    def _pairwise_comparisons(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """Perform pairwise comparisons"""
        pairwise_results = {}
        context_labels = data[context_type].unique()
        
        # Check normality
        normality_result = self._test_normality(data, metric, context_type)
        use_parametric = normality_result['all_groups_normal']
        
        for i, label1 in enumerate(context_labels):
            for label2 in context_labels[i+1:]:
                group1 = data[data[context_type] == label1][metric].dropna()
                group2 = data[data[context_type] == label2][metric].dropna()
                
                comparison_key = f"{label1}_vs_{label2}"
                
                if len(group1) > 0 and len(group2) > 0:
                    try:
                        if use_parametric:
                            # Independent t-test
                            statistic, p_value = stats.ttest_ind(group1, group2)
                            test_name = 'Independent t-test'
                        else:
                            # Mann-Whitney U test
                            statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                            test_name = 'Mann-Whitney U'
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + 
                                            (len(group2)-1)*group2.std()**2) / 
                                           (len(group1)+len(group2)-2))
                        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        pairwise_results[comparison_key] = {
                            'test_name': test_name,
                            'statistic': float(statistic),
                            'p_value': float(p_value),
                            'significant': bool(p_value < self.alpha),
                            'cohens_d': float(cohens_d),
                            'mean_diff': float(group1.mean() - group2.mean()),
                            'group1_mean': float(group1.mean()),
                            'group2_mean': float(group2.mean()),
                            'n1': int(len(group1)),
                            'n2': int(len(group2))
                        }
                    
                    except Exception as e:
                        pairwise_results[comparison_key] = {
                            'error': f'Pairwise test failed: {str(e)}',
                            'n1': int(len(group1)),
                            'n2': int(len(group2))
                        }
        
        return pairwise_results
    
    def _calculate_effect_sizes(self, data: pd.DataFrame, metric: str, context_type: str) -> Dict:
        """Calculate effect sizes for group comparisons"""
        effect_sizes = {}
        context_labels = data[context_type].unique()
        
        # Calculate eta-squared for overall effect
        groups = []
        for label in context_labels:
            group_data = data[data[context_type] == label][metric].dropna()
            groups.extend(group_data.tolist())
        
        if len(groups) > 0:
            overall_mean = np.mean(groups)
            total_ss = sum((x - overall_mean)**2 for x in groups)
            
            between_ss = 0
            for label in context_labels:
                group_data = data[data[context_type] == label][metric].dropna()
                if len(group_data) > 0:
                    group_mean = group_data.mean()
                    between_ss += len(group_data) * (group_mean - overall_mean)**2
            
            eta_squared = between_ss / total_ss if total_ss > 0 else 0
            effect_sizes['eta_squared'] = float(eta_squared)
            
            # Interpret effect size
            if eta_squared < 0.01:
                effect_sizes['interpretation'] = 'negligible'
            elif eta_squared < 0.06:
                effect_sizes['interpretation'] = 'small'
            elif eta_squared < 0.14:
                effect_sizes['interpretation'] = 'medium'
            else:
                effect_sizes['interpretation'] = 'large'
        
        return effect_sizes
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive statistical report"""
        report = "COMPREHENSIVE CONTEXTUAL NETWORK ANALYSIS REPORT\n"
        report += "=" * 60 + "\n\n"
        
        for context_type, results in self.comparison_results.items():
            report += f"CONTEXT TYPE: {context_type.upper().replace('_', ' ')}\n"
            report += "-" * 40 + "\n\n"
            
            # Descriptive statistics
            report += "DESCRIPTIVE STATISTICS:\n"
            for metric, stats in results['descriptive_stats'].items():
                report += f"\n{metric.replace('_', ' ').title()}:\n"
                for context_label, values in stats.items():
                    report += f"  {context_label}: M={values['mean']:.3f}, SD={values['std']:.3f}, n={values['count']}\n"
            
            # Statistical tests
            report += "\nSTATISTICAL TESTS:\n"
            for metric, test_result in results['group_comparisons'].items():
                if 'error' not in test_result:
                    test_name = test_result['test_name']
                    statistic = test_result['statistic']
                    p_value = test_result['p_value']
                    significant = test_result['significant']
                    
                    report += f"\n{metric.replace('_', ' ').title()}:\n"
                    report += f"  {test_name}: χ²/F = {statistic:.3f}, p = {p_value:.3f}"
                    report += f" {'***' if significant else ' (ns)'}\n"
                    
                    # Effect size
                    if metric in results['effect_sizes']:
                        eta_sq = results['effect_sizes'][metric].get('eta_squared', 0)
                        interpretation = results['effect_sizes'][metric].get('interpretation', 'unknown')
                        report += f"  Effect size: η² = {eta_sq:.3f} ({interpretation})\n"
            
            # Significant pairwise comparisons
            report += "\nSIGNIFICANT PAIRWISE COMPARISONS:\n"
            significant_found = False
            for metric, pairwise in results['pairwise_comparisons'].items():
                significant_pairs = []
                for comparison, result in pairwise.items():
                    if result.get('significant', False):
                        p_val = result['p_value']
                        cohens_d = result['cohens_d']
                        mean_diff = result['mean_diff']
                        significant_pairs.append(f"    {comparison}: p={p_val:.3f}, d={cohens_d:.3f}, Δ={mean_diff:.3f}")
                
                if significant_pairs:
                    report += f"\n  {metric.replace('_', ' ').title()}:\n"
                    for pair in significant_pairs:
                        report += pair + "\n"
                    significant_found = True
            
            if not significant_found:
                report += "  No significant pairwise differences found.\n"
            
            report += "\n" + "="*60 + "\n\n"
        
        return report
