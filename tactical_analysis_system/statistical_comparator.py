import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any

class StatisticalComparator:
    """Performs statistical comparisons between contexts"""
    
    def __init__(self):
        self.comparison_results = {}
    
    def compare_contexts(self, results_df: pd.DataFrame) -> Dict:
        """Compare network metrics across different contexts"""
        comparisons = {}
        
        # Compare each context type
        for context_type in results_df['context_type'].unique():
            context_data = results_df[results_df['context_type'] == context_type]
            comparisons[context_type] = self._compare_context_groups(context_data)
        
        self.comparison_results = comparisons
        return comparisons
    
    def _compare_context_groups(self, data: pd.DataFrame) -> Dict:
        """Compare metrics between different context labels"""
        metrics = ['density', 'clustering_coefficient', 'avg_betweenness_centrality',
                  'avg_eigenvector_centrality', 'avg_path_length', 'centralization']
        
        results = {}
        context_labels = data['context_label'].unique()
        
        # Descriptive statistics - convert to JSON serializable format
        desc_stats = data.groupby('context_label')[metrics].agg([
            'count', 'mean', 'std', 'median'
        ]).round(4)
        
        # Convert MultiIndex DataFrame to nested dictionary
        results['descriptive'] = {}
        for context_label in desc_stats.index:
            results['descriptive'][context_label] = {}
            for metric in metrics:
                results['descriptive'][context_label][metric] = {
                    'count': int(desc_stats.loc[context_label, (metric, 'count')]),
                    'mean': float(desc_stats.loc[context_label, (metric, 'mean')]),
                    'std': float(desc_stats.loc[context_label, (metric, 'std')]) if not pd.isna(desc_stats.loc[context_label, (metric, 'std')]) else 0.0,
                    'median': float(desc_stats.loc[context_label, (metric, 'median')])
                }
        
        # Pairwise comparisons
        results['pairwise'] = {}
        for i, label1 in enumerate(context_labels):
            for label2 in context_labels[i+1:]:
                group1 = data[data['context_label'] == label1]
                group2 = data[data['context_label'] == label2]
                
                comparison_key = f"{label1}_vs_{label2}"
                results['pairwise'][comparison_key] = {}
                
                for metric in metrics:
                    values1 = group1[metric].dropna()
                    values2 = group2[metric].dropna()
                    
                    if len(values1) > 0 and len(values2) > 0:
                        # Paired t-test (assuming same matches in both groups)
                        if len(values1) == len(values2):
                            stat, p_value = stats.ttest_rel(values1, values2)
                            test_type = 'paired_ttest'
                        else:
                            stat, p_value = stats.ttest_ind(values1, values2)
                            test_type = 'independent_ttest'
                        
                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(values1)-1)*values1.std()**2 + 
                                            (len(values2)-1)*values2.std()**2) / 
                                           (len(values1)+len(values2)-2))
                        cohens_d = (values1.mean() - values2.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        results['pairwise'][comparison_key][metric] = {
                            'test_type': test_type,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'cohens_d': float(cohens_d),
                            'mean_diff': float(values1.mean() - values2.mean()),
                            'significant': bool(p_value < 0.05),
                            'n1': int(len(values1)),
                            'n2': int(len(values2))
                        }
                    else:
                        # Handle cases with insufficient data
                        results['pairwise'][comparison_key][metric] = {
                            'test_type': 'insufficient_data',
                            'statistic': None,
                            'p_value': None,
                            'cohens_d': None,
                            'mean_diff': None,
                            'significant': False,
                            'n1': int(len(values1)),
                            'n2': int(len(values2))
                        }
        
        return results
    
    def generate_summary_report(self) -> str:
        """Generate a summary report of findings"""
        report = "CONTEXTUAL NETWORK ANALYSIS SUMMARY\n"
        report += "=" * 50 + "\n\n"
        
        for context_type, results in self.comparison_results.items():
            report += f"CONTEXT TYPE: {context_type.upper()}\n"
            report += "-" * 30 + "\n"
            
            # Descriptive summary
            desc = results['descriptive']
            report += "Descriptive Statistics (Mean ± Std):\n"
            for context_label, metrics in desc.items():
                report += f"  {context_label}:\n"
                for metric in ['density', 'clustering_coefficient']:
                    if metric in metrics:
                        mean_val = metrics[metric]['mean']
                        std_val = metrics[metric]['std']
                        count = metrics[metric]['count']
                        report += f"    {metric}: {mean_val:.3f} ± {std_val:.3f} (n={count})\n"
            
            # Significant differences
            report += "\nSignificant Differences (p < 0.05):\n"
            significant_found = False
            for comparison, metrics in results['pairwise'].items():
                significant_metrics = []
                for metric_name, metric_data in metrics.items():
                    if metric_data.get('significant', False):
                        p_val = metric_data['p_value']
                        effect_size = metric_data['cohens_d']
                        significant_metrics.append(f"{metric_name} (p={p_val:.3f}, d={effect_size:.3f})")
                
                if significant_metrics:
                    report += f"  {comparison}:\n"
                    for sig_metric in significant_metrics:
                        report += f"    - {sig_metric}\n"
                    significant_found = True
            
            if not significant_found:
                report += "  No significant differences found.\n"
            
            report += "\n"
        
        return report
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics in a structured format"""
        summary = {}
        
        for context_type, results in self.comparison_results.items():
            summary[context_type] = {
                'total_comparisons': len(results['pairwise']),
                'significant_comparisons': 0,
                'context_labels': list(results['descriptive'].keys()),
                'metrics_analyzed': list(next(iter(results['descriptive'].values())).keys()) if results['descriptive'] else []
            }
            
            # Count significant comparisons
            for comparison, metrics in results['pairwise'].items():
                if any(metric_data.get('significant', False) for metric_data in metrics.values()):
                    summary[context_type]['significant_comparisons'] += 1
        
        return summary
